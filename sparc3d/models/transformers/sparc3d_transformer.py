"""Sparse latent flow transformer for image-conditioned mesh refinement."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: the released sparse inference architecture.
# Modified for Spellcast3D: the zero condition (sparc3d.nn.accel.zero_condition_like) handed to the blocks as such;
# the upsampling residual block normalises before it upsamples (the same rows, a quarter of the float32 memory);
# forward_guided: a conditional and an unconditional prediction sharing the part before the first cross-attention;
# the timestep frequencies copied to the card once a process (sparc3d.nn.accel's sync saving).

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sparc3d.nn import LayerNorm32, accel
from sparc3d.nn.sparse import (
    SparseConv3d,
    SparseDownsample,
    SparseLinear,
    SparseTensor,
    SparseUpsample,
)
from sparc3d.nn.sparse.transformer import ModulatedSparseTransformerCrossBlock
from sparc3d.nn.utils import convert_module_to_f16, zero_module

from .config import Sparc3DTransformerConfig


_FREQUENCIES: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _frequencies(half: int, max_period: int, device: torch.device) -> torch.Tensor:
    """The sinusoids' frequencies computed on the host as the original line does (the same bits) and copied to
    ``device`` once a process instead of every forward (a synchronising copy)."""
    key = (half, max_period, device)
    frequencies = _FREQUENCIES.get(key)
    if frequencies is None:
        with torch.inference_mode(False):       # a normal tensor, usable outside inference mode too
            frequencies = torch.exp(
                -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
            ).to(device)
        _FREQUENCIES[key] = frequencies
    return frequencies


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Args:
            hidden_size: Width of the embedding the MLP produces.
            frequency_embedding_size: Sinusoidal features formed before the MLP.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Create sinusoidal embeddings for scalar timesteps.

        Args:
            timestep: Fractional or integral timesteps ``[B]``.
            dim: Embedding width.
            max_period: Minimum-frequency denominator.

        Returns:
            Embeddings ``[B, dim]``.
        """
        half = dim // 2
        if accel.on("sync"):
            frequencies = _frequencies(half, max_period, timestep.device)
        else:
            frequencies = torch.exp(
                -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
            ).to(timestep.device)
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed ``timestep`` into the transformer hidden width."""
        frequencies = self.timestep_embedding(timestep, self.frequency_embedding_size)
        return self.mlp(frequencies)


class SparseResBlock3d(nn.Module):
    """Timestep-modulated sparse residual block with optional paired resampling."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None = None,
        *,
        downsample: bool = False,
        upsample: bool = False,
    ):
        """
        Args:
            channels: Input feature channels.
            emb_channels: Timestep embedding channels consumed by the modulation.
            out_channels: Output channels; defaults to ``channels``.
            downsample: Halve the spatial resolution across the block.
            upsample: Double the spatial resolution across the block.

        Raises:
            ValueError: Both ``downsample`` and ``upsample`` are set.
        """
        super().__init__()
        if downsample and upsample:
            raise ValueError("cannot downsample and upsample in one block")
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.hidden_channels = self.out_channels // 2
        self.downsample = downsample
        self.upsample = upsample
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(
            self.hidden_channels, elementwise_affine=False, eps=1e-6
        )
        self.conv1 = SparseConv3d(channels, self.hidden_channels, 3)
        self.conv2 = SparseConv3d(self.hidden_channels, self.hidden_channels, 3)
        self.linear = zero_module(SparseLinear(self.hidden_channels, self.out_channels))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.hidden_channels, bias=True),
        )
        self.skip_connection = (
            SparseLinear(channels, self.out_channels)
            if channels != self.out_channels
            else nn.Identity()
        )
        self.updown: nn.Module | None = None
        if downsample:
            self.updown = SparseDownsample(2)
        elif upsample:
            self.updown = SparseUpsample(2)

    def forward(self, x: SparseTensor, embedding: torch.Tensor) -> SparseTensor:
        """Apply residual convolution and optional paired resampling."""
        scale, shift = self.emb_layers(embedding).to(x.dtype).chunk(2, dim=1)
        if (
            accel.on("upsample")
            and self.upsample
            and isinstance(self.skip_connection, SparseLinear)
            and not torch.is_grad_enabled()
        ):
            return self._upsampling_forward(x, scale, shift)
        if self.updown is not None:
            x = self.updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h.replace(F.silu(h.feats))
        return self.linear(h) + self.skip_connection(x)

    def _upsampling_forward(
        self, x: SparseTensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> SparseTensor:
        """``forward`` of an upsampling block with its LayerNorm and SiLU before the upsample.

        The upsample copies each token's row to its voxels, so the LayerNorm and the SiLU
        (row-wise) give the same rows on the tokens, copied after them, as on the copies:
        the voxel-level float32 normalisation (ten bytes a voxel and channel at its peak)
        is never made. The skip projection runs on the voxel rows as before (a GEMM over
        another row count may round differently), its input freed at once; with the item
        ``upsample_token_skip`` (accel.EXTRA) it runs on the tokens instead, a rounding change.
        """
        coords, layout, inverse, voxel_scale = self.updown.mapping(x)
        if accel.extra("upsample_token_skip"):
            skip = self.skip_connection(x).feats[inverse]
        else:
            skip = nn.Linear.forward(self.skip_connection, x.feats[inverse])
        rows = accel.ROW_CHUNK
        normed = torch.empty_like(x.feats)
        for start in range(0, normed.shape[0], rows):
            part = self.norm1(x.feats[start:start + rows])
            normed[start:start + rows] = F.silu(part)
            del part
        voxels = SparseTensor(
            normed[inverse],
            coords,
            shape=x.shape,
            layout=layout,
            scale=voxel_scale,
            spatial_cache=x._spatial_cache,
            trusted=True,
        )
        del normed
        h = self.conv1(voxels)
        del voxels
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h.replace(F.silu(h.feats))
        return self.linear(h) + skip


class Sparc3DTransformer(nn.Module):
    """Released sparse latent flow transformer with a validated narrow config."""

    def __init__(
        self,
        config: Sparc3DTransformerConfig | None = None,
        **config_values: Any,
    ):
        """
        Args:
            config: Validated configuration. Mutually exclusive with ``config_values``.
            **config_values: Configuration fields to validate into a config instead.

        Raises:
            ValueError: Both a config and configuration values were given, or a value
                fails validation.
        """
        super().__init__()
        if config is not None and config_values:
            raise ValueError("pass either config or configuration keyword values")
        self.config = config or Sparc3DTransformerConfig.from_dict(config_values)
        config = self.config
        self.dtype = torch.float16 if config.use_fp16 else torch.float32

        channels = list(config.io_block_channels)
        self.t_embedder = TimestepEmbedder(config.model_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model_channels, 6 * config.model_channels, bias=True),
        )
        self.input_layer = SparseLinear(config.in_channels, channels[0])
        self.input_blocks = nn.ModuleList()
        for current, following in zip(channels, channels[1:] + [config.model_channels]):
            self.input_blocks.extend(
                SparseResBlock3d(current, config.model_channels)
                for _ in range(config.num_io_res_blocks - 1)
            )
            self.input_blocks.append(
                SparseResBlock3d(
                    current,
                    config.model_channels,
                    out_channels=following,
                    downsample=True,
                )
            )

        self.blocks = nn.ModuleList(
            ModulatedSparseTransformerCrossBlock(
                config.model_channels,
                config.cond_channels,
                config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_blocks)
        )

        self.out_blocks = nn.ModuleList()
        previous_channels = [config.model_channels] + list(reversed(channels[1:]))
        for current, previous in zip(reversed(channels), previous_channels):
            self.out_blocks.append(
                SparseResBlock3d(
                    previous * 2,
                    config.model_channels,
                    out_channels=current,
                    upsample=True,
                )
            )
            self.out_blocks.extend(
                SparseResBlock3d(
                    current * 2,
                    config.model_channels,
                    out_channels=current,
                )
                for _ in range(config.num_io_res_blocks - 1)
            )
        self.out_layer = SparseLinear(channels[0], config.out_channels)

        self.initialize_weights()
        if config.use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """Initialize linear, timestep, modulation, and output projections."""

        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize_linear)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """Convert the I/O and transformer torso to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)
        self.dtype = torch.float16

    def forward(
        self, x: SparseTensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> SparseTensor:
        """Predict latent flow for sparse ``x`` under image ``cond``."""
        self._check_inputs(x, timestep, cond)
        h, skips, convolution_embedding, modulation = self._prefix(x, timestep)
        return self._rest(h, skips, convolution_embedding, modulation, cond, x.dtype)

    def forward_guided(
        self,
        x: SparseTensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[SparseTensor, SparseTensor]:
        """``forward(x, timestep, cond)`` and ``forward(x, timestep, negative)``.

        Everything before the first block's cross-attention (the input layer and blocks,
        the timestep embedding, the first block's self-attention) does not see the
        condition, so the pair computes it once: the same two answers, bit for bit.
        """
        self._check_inputs(x, timestep, cond)
        self._check_inputs(x, timestep, negative)
        h, skips, convolution_embedding, modulation = self._prefix(x, timestep)
        h = self.blocks[0].self_attention_part(h, modulation)
        return tuple(
            self._rest(h, skips, convolution_embedding, modulation, c, x.dtype, first_done=True)
            for c in (cond, negative)
        )

    def _check_inputs(
        self, x: SparseTensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> None:
        if x.shape[0] != timestep.shape[0] or x.shape[0] != cond.shape[0]:
            raise ValueError("sparse input, timestep, and condition batches must match")
        if x.feats.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} input channels, "
                f"got {x.feats.shape[1]}"
            )

    def _prefix(
        self, x: SparseTensor, timestep: torch.Tensor
    ) -> tuple[SparseTensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        h = self.input_layer(x).type(self.dtype)
        timestep_embedding = self.t_embedder(timestep)
        convolution_embedding = timestep_embedding.to(self.dtype)
        skips = []
        for block in self.input_blocks:
            h = block(h, convolution_embedding)
            skips.append(h.feats)

        modulation = self.adaLN_modulation(timestep_embedding).to(self.dtype)
        return h, skips, convolution_embedding, modulation

    def _rest(
        self,
        h: SparseTensor,
        skips: list[torch.Tensor],
        convolution_embedding: torch.Tensor,
        modulation: torch.Tensor,
        cond: torch.Tensor,
        out_dtype: torch.dtype,
        first_done: bool = False,
    ) -> SparseTensor:
        """The blocks (the first from its cross-attention on when ``first_done``) and the output blocks."""
        context = cond.to(self.dtype)
        zero_context = accel.is_zero_condition(cond)
        for index, block in enumerate(self.blocks):
            if index == 0 and first_done:
                h = block.context_part(h, modulation, context, zero_context)
            else:
                h = block(h, modulation, context, zero_context=zero_context)

        for block, skip in zip(self.out_blocks, reversed(skips)):
            h = h.replace(torch.cat((h.feats, skip), dim=1))
            h = block(h, convolution_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h.type(out_dtype))
