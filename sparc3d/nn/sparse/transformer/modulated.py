"""Shared-modulation sparse transformer cross-attention block."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: shared timestep modulation with full attention.
# Modified for Spellcast3D: the zero condition's cross-attention row reused (``zero_context``); the forward in two
# parts, before and from the cross-attention (a guided pair shares the first block's first part).

from __future__ import annotations

import torch
from torch import nn

from ... import accel
from ...norm import LayerNorm32
from ..attention import SparseMultiHeadAttention
from ..basic import SparseTensor
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Sparse self/cross-attention block with shared adaptive normalization."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            attention_type="self",
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            ctx_channels=ctx_channels,
            attention_type="cross",
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(1, 6 * channels) / channels**0.5)

    def forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        zero_context: bool = False,
    ) -> SparseTensor:
        """Transform ``x`` with timestep modulation and image context.

        ``zero_context``: ``context`` is all zeros (the unconditional pass), whose
        cross-attention gives one row for every token (``SparseMultiHeadAttention.zero_context``).
        """
        x = self.self_attention_part(x, mod)
        return self.context_part(x, mod, context, zero_context)

    def self_attention_part(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        """The modulated self-attention and its residual: the part that does not see the context."""
        shift_msa, scale_msa, gate_msa = (self.modulation + mod).chunk(6, dim=1)[:3]
        h = self._norm_modulate(x, self.norm1, scale_msa, shift_msa)
        h = self.self_attn(h) * gate_msa
        return x + h

    def context_part(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        zero_context: bool = False,
    ) -> SparseTensor:
        """The cross-attention and the modulated MLP, with their residuals."""
        shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).chunk(6, dim=1)[3:]
        row = self.cross_attn.zero_context_row(x) if zero_context else None
        if row is not None:
            x = x + row
        else:
            h = self._norm_modulate(x, self.norm2)
            x = x + (self.cross_attn.zero_context(h, context) if zero_context
                     else self.cross_attn(h, context))
        h = self._norm_modulate(x, self.norm3, scale_mlp, shift_mlp)
        return x + self.mlp(h) * gate_mlp

    @staticmethod
    def _norm_modulate(
        x: SparseTensor,
        norm: nn.LayerNorm,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> SparseTensor:
        """``norm(x)`` [then ``* (1 + scale) + shift``]: the eager ops, or one fused kernel (accel.FUSED)."""
        if accel.fused_for(x.feats) and len(x.layout) == 1 and x.feats.is_contiguous():
            from ... import fused

            dtype = x.feats.dtype
            return x.replace(fused.norm_modulate(
                x.feats, norm.eps, norm.weight, norm.bias,
                (1 + scale)[0].to(dtype) if scale is not None else None,
                shift[0].to(dtype) if shift is not None else None))
        h = x.replace(norm(x.feats))
        if scale is None:
            return h
        return h * (1 + scale) + shift
