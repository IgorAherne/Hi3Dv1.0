"""Full multi-head attention over sparse variable-length batches."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: full flash attention with spatial RoPE.
# Modified for Spellcast3D: the rotary phases once per pass, q and k written back into the packed qkv, the zero
# condition's cross-attention row once per token count (sparc3d.nn.accel; the same numbers bit for bit).

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ... import accel
from ...rope import RotaryPositionEmbedder
from ..basic import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention


class SparseMultiHeadRMSNorm(nn.Module):
    """Per-head RMS normalization of a ``[N, H, D]`` query or key projection."""

    def __init__(self, dim: int, heads: int):
        """
        Args:
            dim: Channels per head.
            heads: Attention heads, each with its own gain.
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the head dimension in float32 and restore ``x``'s dtype."""
        if accel.on("rms_inplace") and not torch.is_grad_enabled():
            # The same two products, in place.
            y = F.normalize(x.float(), dim=-1)
            return y.mul_(self.gamma).mul_(self.scale).to(x.dtype)
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class SparseMultiHeadAttention(nn.Module):
    """Full flash self- or cross-attention over sparse queries."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        *,
        attention_type: str = "self",
    ):
        """
        Args:
            channels: Query feature channels; must be divisible by ``num_heads``.
            num_heads: Attention heads.
            ctx_channels: Context channels for cross-attention; defaults to
                ``channels``. Ignored for self-attention.
            attention_type: ``"self"`` or ``"cross"``. Self-attention alone carries
                rotary embeddings and query/key RMS normalization, matching the
                parameters the released weights provide.

        Raises:
            ValueError: ``channels`` is not divisible by ``num_heads``, or
                ``attention_type`` is not one of the two supported kinds.
        """
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if attention_type not in {"self", "cross"}:
            raise ValueError("attention_type must be 'self' or 'cross'")
        self.channels = channels
        self.ctx_channels = ctx_channels or channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self._type = attention_type
        if attention_type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=True)
            # Self-attention alone is positional and query/key normalized; the
            # released weights carry no RoPE or RMS parameters on the cross path.
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.rope = RotaryPositionEmbedder(self.head_dim)
        else:
            self.to_q = nn.Linear(channels, channels, bias=True)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=True)
        self.to_out = nn.Linear(channels, channels)
        self._zero_rows: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _linear(
        module: nn.Linear, x: SparseTensor | torch.Tensor
    ) -> SparseTensor | torch.Tensor:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        return module(x)

    def _fused_pre(
        self, x: SparseTensor | torch.Tensor, count: int
    ) -> SparseTensor | torch.Tensor:
        if isinstance(x, SparseTensor):
            return x.reshape(count, self.num_heads, self.head_dim)
        return x.reshape(*x.shape[:2], count, self.num_heads, self.head_dim)

    def forward(
        self,
        x: SparseTensor,
        context: SparseTensor | torch.Tensor | None = None,
    ) -> SparseTensor:
        """Attend from sparse ``x`` to itself or dense ``context``."""
        if self._type == "self":
            qkv = self._fused_pre(self._linear(self.to_qkv, x), 3)
            assert isinstance(qkv, SparseTensor)
            if accel.fused_for(qkv.feats) and self.head_dim == 96 and qkv.feats.is_contiguous():
                from ... import fused

                fused.rope_rms_(qkv.feats, self._rope_phases(qkv), self.q_rms_norm.gamma,
                                self.k_rms_norm.gamma, self.q_rms_norm.scale)
                return self._attend_out(qkv)
            q, k, v = qkv.feats.unbind(dim=1)
            phases = self._rope_phases(qkv) if accel.on("rope_cache") else None
            q, k = self.rope(q, k, qkv.coords[:, 1:], phases)
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            if accel.on("qkv_inplace") and not torch.is_grad_enabled():
                # The values torch.stack would pack, written over the projections.
                qkv.feats[:, 0] = q
                qkv.feats[:, 1] = k
                del q, k, v
            else:
                qkv = qkv.replace(torch.stack((q, k, v), dim=1))
            output = sparse_scaled_dot_product_attention(qkv)
        else:
            if context is None:
                raise ValueError("cross-attention requires context")
            query = self._linear(self.to_q, x)
            assert isinstance(query, SparseTensor)
            query = query.reshape(self.num_heads, self.head_dim)
            kv = self._fused_pre(self._linear(self.to_kv, context), 2)
            output = sparse_scaled_dot_product_attention(query, kv)
        output = output.reshape(-1)
        projected = self._linear(self.to_out, output)
        assert isinstance(projected, SparseTensor)
        return projected

    def _attend_out(self, qkv: SparseTensor) -> SparseTensor:
        output = sparse_scaled_dot_product_attention(qkv).reshape(-1)
        return self._linear(self.to_out, output)

    def _rope_phases(self, qkv: SparseTensor) -> torch.Tensor:
        """The rotary phases of the tokens' coordinates, computed once per spatial cache."""
        key = f"rope_phases_{self.head_dim}"
        phases = qkv.get_spatial_cache(key)
        if phases is None:
            phases = self.rope.phases(qkv.coords[:, 1:])
            qkv.register_spatial_cache(key, phases)
        return phases

    def _zero_key(self, x: SparseTensor) -> tuple:
        return (x.feats.shape[0], x.feats.dtype, x.device, id(self.to_kv), id(self.to_out))

    def zero_context_row(self, x: SparseTensor) -> torch.Tensor | None:
        """The output row ``zero_context`` found for ``x``'s token count, or None yet."""
        return self._zero_rows.get(self._zero_key(x))

    def zero_context(self, h: SparseTensor, context: torch.Tensor) -> SparseTensor:
        """Cross-attention to an all-zero ``context``, ``h`` the normalised queries.

        Zero context tokens project to ``to_kv``'s bias: identical keys, so every query
        weighs the identical values uniformly and every token gets the same output row.
        This runs the full path and keeps that row (when every token's row is equal) for
        ``zero_context_row``: ``[1, C]``, which the residual add broadcasts.
        """
        output = self(h, context)
        first = output.feats[:1]
        if torch.equal(output.feats, first.expand_as(output.feats)):
            self._zero_rows = {self._zero_key(h): first.clone()}
        return output
