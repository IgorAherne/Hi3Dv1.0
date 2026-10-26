"""Attention kernels for the sparse transformer and the decoder's windowed attention.

Modified for Spellcast3D: flash-attn where it is installed and the card is sm_80 or newer, else PyTorch's
scaled_dot_product_attention on the same packed tensors (the numbers differ only by kernel rounding).
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=None)
def _flash_on(device_index: int):
    if torch.cuda.get_device_capability(device_index) < (8, 0):
        return None
    try:
        import flash_attn
    except ImportError:
        return None
    return flash_attn


def flash_attn_for(device: torch.device):
    """flash_attn, or None where PyTorch's attention has to stand in."""
    if device.type != "cuda":
        return None
    return _flash_on(device.index if device.index is not None else torch.cuda.current_device())


def sdpa_spans(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_q: list[int], cu_k: list[int],
               scale: float | None = None) -> torch.Tensor:
    """Variable-length attention over [N, H, D] spans with PyTorch's kernel, one span at a time."""
    out = torch.empty_like(q)
    for i in range(len(cu_q) - 1):
        qs, ks = slice(cu_q[i], cu_q[i + 1]), slice(cu_k[i], cu_k[i + 1])
        out[qs] = F.scaled_dot_product_attention(
            q[qs].transpose(0, 1)[None], k[ks].transpose(0, 1)[None], v[ks].transpose(0, 1)[None],
            scale=scale)[0].transpose(0, 1)
    return out


def sdpa_windows(qkv: torch.Tensor, window: int, scale: float | None = None) -> torch.Tensor:
    """Attention inside consecutive windows of equal length: qkv [N, 3, H, D] with N a multiple of window."""
    n, _, heads, dim = qkv.shape
    w = qkv.view(n // window, window, 3, heads, dim).permute(2, 0, 3, 1, 4)    # 3, B, H, W, D
    out = F.scaled_dot_product_attention(w[0], w[1], w[2], scale=scale)        # B, H, W, D
    return out.transpose(1, 2).reshape(n, heads, dim)
