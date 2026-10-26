"""Sparse PTv3 building blocks: norms, residual blocks, serialized attention, resampling.

Module and parameter names are load-bearing: they define the checkpoint key layout.
"""

# Adapted from Pointcept / Point Transformer V3 (Xiaoyang Wu et al.) under the
# MIT License. Modified for SPARC3D: specialized for the sparse mesh
# autoencoder, with the training paths and drop-path removed.
# Modified for Spellcast3D: PyTorch's attention where flash-attn is missing or the card predates sm_80; an
# opt-in GroupNorm whose moments are summed in float64 in a canonical order (exact_group_norm), for the
# slab-streamed decode.

from __future__ import annotations

import contextlib
import warnings

import torch
from torch import nn

from sparc3d.nn.sparse.attention.backend import flash_attn_for, sdpa_spans, sdpa_windows
from sparc3d.ops import sparse_backend
from sparc3d.structures.point import Point
from sparc3d.structures.point_modules import PointModule, PointSequential
from sparc3d.utils.misc import batch2offset, offset2bincount
from sparc3d.utils.precision import force_fp32


_RESNET_SKIP_POINTS = 12_000_000
"""Default ``SparseResnetBlock.skip_points``: above this token count the residual
blocks pass the point through untouched and warn. The 1024 path can exceed what a
full residual pass fits in memory, and the released weights were measured with this
guard active, so it is reproduced behavior rather than a tuning knob.

Deliberately the same number as ``pipelines.sparc3d_pipeline.DECODE_FEATURE_CEILING``,
but a different policy: this one degrades a single block, while the pipeline raises
when a *completed* decode is too large to feed another pass."""

_GROUP_NORM_SPLIT = 8
"""Groups normalized per iteration. Group statistics are independent, so this only
bounds the temporaries, not the result."""

MOMENT_CHUNK_ROWS = 1 << 15
"""Rows per float64 partial sum in :func:`moment_sums`."""

_exact_moments = False


@contextlib.contextmanager
def exact_group_norm():
    """Inside the block, PointGroupNorm sums its moments in float64, in one canonical order.

    The default float32 ``scatter_add`` runs as atomics in no fixed order, so its sums vary
    in the last bits from run to run. Here the points are taken in (x, y, z) order and summed
    :data:`MOMENT_CHUNK_ROWS` at a time (:func:`moment_sums`), so the result is the same for
    any order of the rows: the slab-streamed decode sums the same chunks.
    """
    global _exact_moments
    previous, _exact_moments = _exact_moments, True
    try:
        yield
    finally:
        _exact_moments = previous


def moment_chunk_sums(chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel float64 sums of one chunk of rows and of their float32 squares ``[C]``."""
    chunk = chunk.float()
    return chunk.double().sum(dim=0), chunk.square().double().sum(dim=0)


def moment_sums(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel float64 sums of ``feat`` and of its float32 square: chunk by chunk, in row order."""
    sums = torch.zeros(feat.shape[1], dtype=torch.float64, device=feat.device)
    sums_sq = torch.zeros_like(sums)
    for start in range(0, feat.shape[0], MOMENT_CHUNK_ROWS):
        chunk_sums, chunk_sums_sq = moment_chunk_sums(feat[start:start + MOMENT_CHUNK_ROWS])
        sums += chunk_sums
        sums_sq += chunk_sums_sq
    return sums, sums_sq


def canonical_order(point: Point) -> torch.Tensor:
    """Row order of ``point`` by (x, y, z) grid coordinate."""
    if "sparse_conv_feat" in point:
        coords = point.sparse_conv_feat.indices[:, 1:].long()
    else:
        coords = point.grid_coord.long()
    return torch.argsort((coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2])


def group_sums(channel_sums: torch.Tensor, groups: int) -> torch.Tensor:
    """Per-channel float64 sums gathered into ``[1, groups]`` float32 group sums."""
    return channel_sums.view(groups, -1).sum(dim=1).float().view(1, groups)


class Swish(nn.Module):
    """``x * sigmoid(x)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Swish activation elementwise."""
        return x * torch.sigmoid(x)


class PointGroupNorm(PointModule):
    """Group normalization over the points of each batch element of a sparse point cloud."""

    def __init__(
        self,
        channels: int,
        group: int = 32,
        min_group_channels: int = 4,
        eps: float = 1e-6,
    ):
        """
        Args:
            channels: Feature channels.
            group: Requested number of groups; reduced when ``channels`` is too small
                to give every group ``min_group_channels`` channels.
            min_group_channels: Minimum channels per group.
            eps: Variance floor.

        Raises:
            ValueError: ``channels`` is not divisible by the resolved group count.
        """
        super().__init__()
        self.eps = eps
        self.group = group
        self.channels = channels
        self.min_group_channels = min_group_channels

        if self.min_group_channels * self.group > channels:
            self.group = channels // self.min_group_channels
        if self.group < 1 or channels % self.group != 0:
            raise ValueError(
                f"channels {channels} must be divisible by group {self.group}"
            )
        self.channels_per_group = channels // self.group

        self.weights = nn.Parameter(torch.ones(1, channels))
        self.bias = nn.Parameter(torch.zeros(1, channels))

    @force_fp32(mode="convert_back")
    def forward(self, point: Point) -> Point:
        """Normalize ``point.feat`` per batch element and group, in float32."""
        feat = point.feat
        batch = point.batch.detach().clone()
        N, C = feat.shape
        B = int(batch[-1] + 1)

        G = self.group
        CP = self.channels_per_group
        feat_group = feat.view(N, G, CP)
        feat_norm_all = feat.new_zeros((N, G, CP))

        exact = _exact_moments and B == 1
        if exact:
            exact_sums, exact_sums_sq = (group_sums(s, G) for s in moment_sums(feat[canonical_order(point)]))

        for start in range(0, G, _GROUP_NORM_SPLIT):
            end = min(start + _GROUP_NORM_SPLIT, G)

            feat_slice = feat_group[:, start:end, :]
            if exact:
                sums = exact_sums[:, start:end].clone()
                sums_sq = exact_sums_sq[:, start:end].clone()
            else:
                sum_feat = feat_slice.sum(dim=2)
                sum_feat_sq = feat_slice.square().sum(dim=2)

                # Per-(batch, group) moments, accumulated over the points of each element.
                batch_idx = batch.view(-1, 1).expand(-1, end - start).long()
                sums = feat.new_zeros((B, end - start)).scatter_add(0, batch_idx, sum_feat)
                sums_sq = feat.new_zeros((B, end - start)).scatter_add(
                    0, batch_idx, sum_feat_sq
                )
                del sum_feat, sum_feat_sq

            counts = torch.bincount(batch, minlength=B).float().view(B, 1) * CP
            mean = sums.div_(counts + self.eps)
            var = sums_sq.div_(counts + self.eps).sub_(mean.square())
            inv_std = torch.rsqrt(var + self.eps)

            mean_exp = mean.index_select(0, batch).unsqueeze(2)
            inv_std_exp = inv_std.index_select(0, batch).unsqueeze(2)
            feat_norm_all[:, start:end, :] = feat_slice.sub(mean_exp).mul_(inv_std_exp)

            del feat_slice, sums, sums_sq
            del mean, var, inv_std, mean_exp, inv_std_exp

        feat_norm_all = feat_norm_all.view(N, C)
        feat_norm_all.mul_(self.weights).add_(self.bias)

        point.update_feat(feat_norm_all)
        return point


class LayerNormFP32(nn.LayerNorm):
    """Layer norm that always computes in float32, then restores the input dtype."""

    @force_fp32(mode="convert_back")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x`` in float32."""
        return super().forward(x)


class MLP(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
    ):
        """
        Args:
            in_channels: Input channels.
            hidden_channels: Width of the hidden layer; defaults to ``in_channels``.
            out_channels: Output channels; defaults to ``in_channels``.
            act_layer: Activation constructed between the two linear layers.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` through the hidden layer and back."""
        return self.fc2(self.act(self.fc1(x)))


class SparseClsHead(PointModule):
    """Per-voxel classification head, used to predict octree occupancy on upsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Input feature channels.
            out_channels: Logits emitted per voxel; eight for octree occupancy.
        """
        super().__init__()
        self.fc = PointSequential(
            PointGroupNorm(in_channels),
            Swish(),
            sparse_backend.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

    @force_fp32(mode="convert")
    def forward(self, point: Point) -> Point:
        """Return a point whose ``feat`` holds the per-voxel occupancy logits."""
        # Wrap only the tensors the head reads; cloning the whole point here would
        # keep every intermediate of the calling stage alive.
        head_point = Point(
            dict(
                feat=point.feat,
                sparse_conv_feat=point.sparse_conv_feat,
                batch=point.batch,
            )
        )
        head_point.is_sparsified = True
        return self.fc(head_point)


class ConvOut(PointModule):
    """Output projection that splits off the eight per-corner SDF sign logits."""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Input feature channels.
            out_channels: Output channels, the first eight being the SDF sign logits.
        """
        super().__init__()
        self.conv = PointSequential(
            PointGroupNorm(in_channels),
            Swish(),
            sparse_backend.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    @force_fp32(mode="convert")
    def forward(self, point: Point) -> Point:
        """Split the projection into ``point["sdf_sign"]`` and the remaining features."""
        point = self.conv(point)
        point["sdf_sign"] = point.feat[:, :8]
        point.update_feat(point.feat[:, 8:])
        return point


class SparseZToF(PointModule):
    """Project latent channels back to decoder features."""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Latent channels arriving from the transformer.
            out_channels: Decoder feature channels at the coarsest stage.
        """
        super().__init__()
        self.conv = PointSequential(
            sparse_backend.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            )
        )

    @force_fp32(mode="enforce")
    def forward(self, point: Point) -> Point:
        """Project latent channels up to the decoder's feature width."""
        return self.conv(point)


class SparseResnetBlock(PointModule):
    """Pre-norm residual block of two sparse 3x3x3 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_points: int = _RESNET_SKIP_POINTS,
    ):
        """
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            skip_points: Token count above which the block passes its input through
                unchanged. Defaults to the product's own guard.
        """
        super().__init__()
        self.skip_points = skip_points
        self.block = PointSequential(
            PointGroupNorm(in_channels),
            Swish(),
            sparse_backend.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            PointGroupNorm(out_channels),
            Swish(),
            sparse_backend.Conv3d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

        # A PointSequential shortcut would alias and then overwrite point.feat.
        if in_channels != out_channels:
            self.res_conv = nn.Linear(in_channels, out_channels)
        else:
            self.res_conv = None

    def forward(self, point: Point) -> Point:
        """Add the residual branch to ``point``, or pass it through when oversized."""
        if point.feat.shape[0] > self.skip_points:
            warnings.warn(
                f"SparseResnetBlock skipped: {point.feat.shape[0]} points exceeds the "
                f"skip_points budget of {self.skip_points}. The block is a no-op, so "
                "the output is the input; raise skip_points to run it.",
                stacklevel=2,
            )
            return point
        if self.res_conv is not None:
            shortcut = self.res_conv(point.feat)
        else:
            shortcut = point.feat
        point.update_feat(shortcut + self.block(point).feat)
        return point


class SerializedAttention(PointModule):
    """Windowed self-attention over patches of a space-filling-curve ordering."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        order_index: int = 0,
    ):
        """
        Args:
            channels: Feature channels; must be divisible by ``num_heads``.
            num_heads: Attention heads.
            patch_size: Points per attention window.
            order_index: Which of the point's serialized orders to attend along.

        Raises:
            ValueError: ``channels`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels {channels} must be divisible by num_heads {num_heads}"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads)**-0.5
        self.order_index = order_index
        self.patch_size = patch_size

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=True)
        self.proj = torch.nn.Linear(channels, channels)

    @torch.no_grad()
    def get_padding_and_inverse(self, point: Point):
        """Return, and cache on ``point``, the patch padding maps and sequence bounds.

        Each batch element is padded up to a whole number of patches by repeating its
        last full patch, so attention never needs a mask.

        Returns:
            ``(pad, unpad, cu_seqlens)``: the padded gather index, its inverse, and
            the cumulative patch boundaries flash attention expects.
        """
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # Only pad an element when it has more points than one patch.
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                # yapf: disable
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]] = \
                        pad[_offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1] - self.patch_size]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                # yapf: enable
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point: Point) -> Point:
        """Attend within each patch of the serialized ordering and project the result."""
        H = self.num_heads
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]

        packed = qkv.half().reshape(-1, 3, H, C // H)
        flash_attn = flash_attn_for(packed.device)
        if flash_attn is not None:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                packed,
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=0,
                softmax_scale=self.scale,
            )
        elif packed.shape[0] % self.patch_size == 0 and packed.shape[0] == (len(cu_seqlens) - 1) * self.patch_size:
            feat = sdpa_windows(packed, self.patch_size, self.scale)
        else:
            bounds = cu_seqlens.tolist()
            q, k, v = packed.unbind(dim=1)
            feat = sdpa_spans(q, k, v, bounds, bounds, self.scale)
        feat = feat.reshape(-1, C).to(qkv.dtype)[inverse]

        point.feat = self.proj(feat)
        return point


class SparseAttentionBlock(PointModule):
    """Serialized attention block with a sparse conditional position encoding and MLP."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 48,
        mlp_ratio: float = 4.0,
        order_index: int = 0,
    ):
        """
        Args:
            channels: Feature channels, preserved across the block.
            num_heads: Attention heads.
            patch_size: Points per attention window.
            mlp_ratio: Hidden width of the MLP as a multiple of ``channels``.
            order_index: Which of the point's serialized orders to attend along.
        """
        super().__init__()
        self.channels = channels

        norm_layer = LayerNormFP32
        act_layer = nn.GELU

        self.cpe = PointSequential(
            sparse_backend.Conv3d(channels, channels, kernel_size=3, bias=True),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            order_index=order_index,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
            )
        )

    def forward(self, point: Point) -> Point:
        """Apply the position encoding, attention and MLP residual branches in turn."""
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + point.feat
        # Everything above touched only feat; resync the sparse view once.
        point.sparse_conv_feat = sparse_backend.replace_feats(
            point.sparse_conv_feat, point.feat
        )
        return point


class SerializedUnpooling(PointModule):
    """Double the grid resolution, keeping the children the occupancy head predicts."""

    def __init__(self, channels: int):
        """
        Args:
            channels: Feature channels, preserved across the resolution doubling.
        """
        super().__init__()
        self.cls_head = SparseClsHead(channels, 8)
        self.proj = nn.Linear(channels, channels * 8, bias=False)
        self.inv_grid_coord = nn.Parameter(
            torch.tensor(
                [
                    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                ]
            ),
            requires_grad=False,
        )

    def forward(self, point: Point) -> Point:
        """Split each voxel into eight children and keep the ones predicted occupied."""
        feat = self.proj(point.feat)
        pred_octree = self.cls_head(point).feat

        m, c = feat.shape[0], feat.shape[1] // 8
        mask = pred_octree.flatten() > 0

        feat = feat.view(m * 8, c)[mask]
        grid_coord = point.grid_coord.unsqueeze(1) * 2 + self.inv_grid_coord.unsqueeze(0)
        grid_coord = grid_coord.view(-1, 3)[mask]
        batch = point.batch.unsqueeze(1).expand(-1, 8).flatten()[mask]

        point = point.clone(
            feat=feat,
            grid_coord=grid_coord,
            resolution=point.resolution * 2,
            batch=batch,
            offset=batch2offset(batch),
        )
        point.sparsify()
        return point


class Upsample(PointModule):
    """Stage-boundary upsampling by serialized unpooling."""

    def __init__(self, channels: int):
        """
        Args:
            channels: Feature channels, preserved across the upsampling.
        """
        super().__init__()
        self.upsample = SerializedUnpooling(channels)

    def forward(self, point: Point) -> Point:
        """Double the grid resolution of ``point``."""
        # The point count grows up to eightfold here; release the freed stage first.
        torch.cuda.empty_cache()
        return self.upsample(point)
