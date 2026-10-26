"""Variable-length full flash attention for sparse feature sequences."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: variable-length flash attention only.
# Modified for Spellcast3D: PyTorch's attention where flash-attn is missing or the card predates sm_80; an
# engine's self-attention kernel for one sequence (sparc3d.nn.accel.self_attention); the cumulative lengths kept
# per pass (sparc3d.nn.accel's sync saving).

from __future__ import annotations

import torch

from ... import accel
from ..basic import SparseTensor
from .backend import flash_attn_for, sdpa_spans


def _lengths(tensor: SparseTensor) -> list[int]:
    return [span.stop - span.start for span in tensor.layout]


def _cumulative(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0, *lengths], device=device, dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)


def _cumulative_of(tensor: SparseTensor, lengths: list[int], key: str) -> torch.Tensor:
    """``_cumulative(lengths, tensor.device)``, kept in the tensor's spatial cache with the sync saving on: a pass's
    layout and context length are fixed, so the host copy (a synchronisation) runs once a pass, not every call."""
    if not accel.on("sync"):
        return _cumulative(lengths, tensor.device)
    shape = tuple(lengths)
    cached = tensor.get_spatial_cache(key)
    if cached is None or cached[0] != shape:
        cached = (shape, _cumulative(lengths, tensor.device))
        tensor.register_spatial_cache(key, cached)
    return cached[1]


def sparse_scaled_dot_product_attention(
    query: SparseTensor, kv: torch.Tensor | None = None
) -> SparseTensor:
    """Apply variable-length full flash attention over sparse queries.

    Args:
        query: Packed sparse ``qkv`` of shape ``[N, 3, H, D]`` for self-attention,
            or sparse queries ``[N, H, D]`` when ``kv`` is given.
        kv: Dense packed keys and values ``[B, L, 2, H, D]`` for cross-attention;
            ``None`` selects self-attention over the packed ``query``.

    Returns:
        Sparse output over the query's coordinates.

    Raises:
        TypeError: ``query`` is not sparse.
        ValueError: ``kv`` does not share the query's batch size.
    """
    if not isinstance(query, SparseTensor):
        raise TypeError("Sparc3D sparse attention requires sparse queries")
    query_lengths = _lengths(query)
    if kv is None and accel.ENABLED and accel.self_attention is not None and len(query_lengths) == 1:
        q, k, v = query.feats.unbind(dim=1)
        output = accel.self_attention(q[None], k[None], v[None])
        if output is not None:
            return query.replace(output.reshape(q.shape))
    flash_attn = flash_attn_for(query.device)
    if flash_attn is None:
        return query.replace(_sdpa(query.feats, kv, query_lengths))
    cumulative_query = _cumulative_of(query, query_lengths, "cu_seqlens_q")

    if kv is None:
        output = flash_attn.flash_attn_varlen_qkvpacked_func(
            query.feats, cumulative_query, max(query_lengths)
        )
        return query.replace(output)

    batch, length = kv.shape[:2]
    if batch != query.shape[0]:
        raise ValueError("query and context batch sizes must match")
    key_lengths = [length] * batch
    output = flash_attn.flash_attn_varlen_kvpacked_func(
        query.feats,
        kv.reshape(batch * length, *kv.shape[2:]),
        cumulative_query,
        _cumulative_of(query, key_lengths, "cu_seqlens_kv"),
        max(query_lengths),
        max(key_lengths),
    )
    return query.replace(output)


def _sdpa(feats: torch.Tensor, kv: torch.Tensor | None, query_lengths: list[int]) -> torch.Tensor:
    cu_q = [0]
    for length in query_lengths:
        cu_q.append(cu_q[-1] + length)
    if kv is None:
        q, k, v = feats.unbind(dim=1)
        return sdpa_spans(q, k, v, cu_q, cu_q)
    batch, length = kv.shape[:2]
    if batch != len(query_lengths):
        raise ValueError("query and context batch sizes must match")
    k, v = kv.reshape(batch * length, *kv.shape[2:]).unbind(dim=1)
    return sdpa_spans(feats, k, v, cu_q, list(range(0, (batch + 1) * length, length)))
