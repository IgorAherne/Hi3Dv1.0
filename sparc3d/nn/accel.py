"""Switches and hooks of the sampler's speed and memory work.

Written for Spellcast3D. ``SPARC3D_FAST`` (environment, read at import; ``ENABLED`` changes it in a running
process) turns the work on or off:

    "1", "on", "true", "yes", unset   on: the sampler's per-pass caches, sync-free bookkeeping and leaner memory
                                      (the original path's numbers bit for bit, given the downsample's sums in one
                                      fixed order: DETERMINISTIC), plus whatever hooks an engine installs below
    "0", "off", "false", "no"         the original code path, line for line

``SPARC3D_FUSED`` ("1"/"on"; off when unset; ``FUSED`` in a running process): the blocks' elementwise chains as
fused Triton kernels (sparc3d.nn.fused) - a change of rounding, not bit for bit, so off unless asked for. Where a
fused kernel runs, the OFF items of the eager chain it replaces do nothing: the self-attention's fused path ignores
rope_cache (its rotary phases always come from the pass's cache), rms_inplace, rope_inplace and qkv_inplace, and
the fused norm modulation ignores broadcast.

Per-pass caches (the sample's coordinates are fixed through a pass, and the sample tensors of a pass share one
spatial cache): the downsample's mapping, spconv's index pairs and grid, the rotary phases, flash-attn's cumulative
sequence lengths, and the zero condition's cross-attention row (``to_kv`` of zeros is its bias, so every token attends uniformly to one value row
whatever its query: the row ``to_out`` makes of it is computed by the full path once per token count and reused).

Hooks an engine may set (``None``: the built-in path):

    self_attention(q, k, v) -> out    the full self-attention of one sequence: q, k, v ``[1, N, H, D]`` (last
                                      dimension contiguous), ``out`` the same shape and dtype, or None to decline
                                      (flash-attn / SDPA then run). Offered only single-sequence self-attention;
                                      cross-attention keeps flash-attn / SDPA. The transformer's own candidate:
                                      sparc3d.nn.sparse.attention.int8_attention.install().
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sparc3d.nn import fused


def _from_env() -> bool:
    return os.environ.get("SPARC3D_FAST", "").strip().lower() not in {"0", "off", "false", "no"}


ENABLED: bool = _from_env()

OFF: set[str] = set()
"""Items turned off one by one while ENABLED (checks): sync, broadcast, geometry, downsample, rope_cache,
qkv_inplace, rms_inplace, rope_inplace, gelu_inplace, upsample, zero_row, shared_prefix, host_times."""


def on(item: str) -> bool:
    """Whether the saving ``item`` runs (ENABLED and not in OFF)."""
    return ENABLED and item not in OFF


EXTRA: set[str] = set()
"""Items that change rounding, off unless named here: upsample_token_skip."""


def extra(item: str) -> bool:
    """Whether the opt-in ``item`` runs (ENABLED and in EXTRA)."""
    return ENABLED and item in EXTRA


DETERMINISTIC: bool = True
"""The downsample's mean summed in one fixed order (SparseDownsample._serial_mean) instead of CUDA's fp16 atomics,
whose order - and so the last bits of the tokens, which every step then amplifies - changes from run to run: the
original sampler is not reproducible (two forwards on the same input differ by ~0.3 % relative L2). With it, a pass
gives the same bits every time, and every other saving here gives the bits of the original path with it."""

FUSED: bool = os.environ.get("SPARC3D_FUSED", "").strip().lower() in {"1", "on", "true", "yes"}


def fused_for(x: torch.Tensor) -> bool:
    """Whether the fused kernels take a block's work on ``x`` (the switches on, the kernels usable here)."""
    if not (ENABLED and FUSED):
        return False
    return fused.usable(x)


self_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None

ROW_CHUNK = 16384
"""Rows per piece where a row-wise step runs in pieces to bound its float32 temporaries."""

ZERO_CONDITION = "_sparc3d_zero_condition"
"""Attribute a zero condition tensor carries (``zero_condition_like``): the transformer then knows it is all zeros
without looking at it."""


def zero_condition_like(condition: torch.Tensor) -> torch.Tensor:
    """``torch.zeros_like(condition)``, marked as the all-zero (unconditional) condition."""
    zeros = torch.zeros_like(condition)
    setattr(zeros, ZERO_CONDITION, True)
    return zeros


def is_zero_condition(condition: torch.Tensor) -> bool:
    """Whether ``condition`` was made by ``zero_condition_like`` (and the shortcuts are on)."""
    return on("zero_row") and bool(getattr(condition, ZERO_CONDITION, False))
