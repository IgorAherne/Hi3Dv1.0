"""Int8 Q.K^T, fp16 P.V self-attention for head size 96 (Triton), for ``sparc3d.nn.accel.self_attention``.

Written for Spellcast3D. The transformer's self-attention over all tokens of a prop is the largest part of a
sampling step at the finer passes, and flash-attn already runs it at the card's fp16 tensor peak (fp32
accumulation). This kernel follows SageAttention's recipe at the transformer's own head size, 96, without padding
it to 128 (SageAttention's kernels take 64 or 128): q and k quantised to int8 with one scale per token and head
(k's mean over the tokens subtracted first, which leaves every softmax row unchanged; q carrying the softmax scale
and log2(e)), the logits from int8 tensor-core products summed exactly in int32, the softmax in exp2, P rounded to
fp16 and multiplied by v in fp16, each 64-key block's product summed in fp16 and added to an fp32 accumulator
("fp16+fp32", as SageAttention's sm_8x kernel; ``pv_f16=False`` sums in fp32). The 96 channels run as a 64- and a
32-wide piece (Triton's tiles are powers of two).

It changes the numbers (the int8 logits, the fp16 partial sums): a different rounding, judged end to end against
the samplers' rounding-only control before an engine turns it on. Usable on CUDA cards from sm_80 (Triton's int8
mma.sync) with Triton installed; ``install`` runs ``self_check`` first and leaves the hook unset when it fails.

    install(pv_f16=True, device=None) -> bool    the hook set (True) or not (False, the reason in ``state()``)
    attention(q, k, v, pv_f16=True)               q, k, v [N, H, 96] or [1, N, H, 96] (last dimension contiguous)
"""

from __future__ import annotations

import warnings

import torch

from ... import accel

HEAD_DIM = 96
MIN_TOKENS = 64
LOG2E = 1.4426950408889634
BLOCK_M, BLOCK_N, WARPS, STAGES = 128, 64, 4, 3       # the fastest of a tile sweep on an RTX 3060 (1.8x flash-attn)
QUANT_ROWS = 64

_kernels: dict | None = None
_state: dict = {"ok": None, "why": ""}


def _build() -> dict:
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def quant_kernel(X, Q, S, M, n_rows, stride_xn, stride_xh, mult,
                     SUB_MEAN: tl.constexpr, BLOCK_R: tl.constexpr):
        # one head's rows: (x * mult [- mean]) -> int8 [H, N, 96] + one fp32 scale per token
        pid = tl.program_id(0)
        h = tl.program_id(1)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < n_rows
        ca = tl.arange(0, 64)
        cb = 64 + tl.arange(0, 32)
        base = X + rows[:, None].to(tl.int64) * stride_xn + h * stride_xh
        xa = tl.load(base + ca[None, :], mask=rmask[:, None], other=0.0).to(tl.float32) * mult
        xb = tl.load(base + cb[None, :], mask=rmask[:, None], other=0.0).to(tl.float32) * mult
        if SUB_MEAN:
            xa = xa - tl.load(M + h * 96 + ca)[None, :]
            xb = xb - tl.load(M + h * 96 + cb)[None, :]
        amax = tl.maximum(tl.maximum(tl.max(tl.abs(xa), axis=1), tl.max(tl.abs(xb), axis=1)), 1e-30)
        inv = 127.0 / amax
        qa = libdevice.rint(xa * inv[:, None])
        qb = libdevice.rint(xb * inv[:, None])
        obase = Q + (h.to(tl.int64) * n_rows + rows[:, None]) * 96
        tl.store(obase + ca[None, :], qa.to(tl.int8), mask=rmask[:, None])
        tl.store(obase + cb[None, :], qb.to(tl.int8), mask=rmask[:, None])
        tl.store(S + h.to(tl.int64) * n_rows + rows, amax / 127.0, mask=rmask)

    @triton.jit
    def attn_kernel(Q8, K8, QS, KS, V, O, n, stride_vn, stride_vh, stride_on, stride_oh,
                    PV_F16: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        h = tl.program_id(1).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        ca = tl.arange(0, 64)
        cb = 64 + tl.arange(0, 32)
        mmask = offs_m < n
        qbase = Q8 + (h * n + offs_m[:, None]) * 96
        qa = tl.load(qbase + ca[None, :], mask=mmask[:, None], other=0)
        qb = tl.load(qbase + cb[None, :], mask=mmask[:, None], other=0)
        qs = tl.load(QS + h * n + offs_m, mask=mmask, other=0.0)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_a = tl.zeros([BLOCK_M, 64], dtype=tl.float32)
        acc_b = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        for start in range(0, n, BLOCK_N):
            kn = start + offs_n
            nmask = kn < n
            kbase = K8 + (h * n + kn[None, :]) * 96
            ka = tl.load(kbase + ca[:, None], mask=nmask[None, :], other=0)
            kb = tl.load(kbase + cb[:, None], mask=nmask[None, :], other=0)
            ks = tl.load(KS + h * n + kn, mask=nmask, other=0.0)
            s = tl.dot(qa, ka, out_dtype=tl.int32)
            s = tl.dot(qb, kb, s, out_dtype=tl.int32)
            qk = s.to(tl.float32) * qs[:, None] * ks[None, :]
            qk = tl.where(nmask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc_a = acc_a * alpha[:, None]
            acc_b = acc_b * alpha[:, None]
            vbase = V + kn[:, None].to(tl.int64) * stride_vn + h * stride_vh
            va = tl.load(vbase + ca[None, :], mask=nmask[:, None], other=0.0)
            vb = tl.load(vbase + cb[None, :], mask=nmask[:, None], other=0.0)
            p16 = p.to(tl.float16)
            if PV_F16:
                acc_a += tl.dot(p16, va, out_dtype=tl.float16).to(tl.float32)
                acc_b += tl.dot(p16, vb, out_dtype=tl.float16).to(tl.float32)
            else:
                acc_a = tl.dot(p16, va, acc_a)
                acc_b = tl.dot(p16, vb, acc_b)
            m_i = m_new
        acc_a = acc_a / l_i[:, None]
        acc_b = acc_b / l_i[:, None]
        obase = O + offs_m[:, None].to(tl.int64) * stride_on + h * stride_oh
        tl.store(obase + ca[None, :], acc_a.to(O.dtype.element_ty), mask=mmask[:, None])
        tl.store(obase + cb[None, :], acc_b.to(O.dtype.element_ty), mask=mmask[:, None])

    _kernels = {"quant": quant_kernel, "attn": attn_kernel, "cdiv": triton.cdiv}
    return _kernels


def _quantize(x: torch.Tensor, mult: float = 1.0, mean: torch.Tensor | None = None):
    k = _build()
    n, heads, _ = x.shape
    q = torch.empty((heads, n, HEAD_DIM), dtype=torch.int8, device=x.device)
    s = torch.empty((heads, n), dtype=torch.float32, device=x.device)
    k["quant"][(k["cdiv"](n, QUANT_ROWS), heads)](
        x, q, s, mean if mean is not None else s, n, x.stride(0), x.stride(1), float(mult),
        SUB_MEAN=mean is not None, BLOCK_R=QUANT_ROWS, num_warps=4)
    return q, s


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pv_f16: bool = True) -> torch.Tensor:
    """softmax(q k^T / sqrt(96)) v over one sequence; q, k, v fp16 [N, H, 96] or [1, N, H, 96], the last
    dimension contiguous; the answer in q's shape and dtype."""
    batched = q.dim() == 4
    if batched:
        q, k, v = q[0], k[0], v[0]
    n, heads, d = q.shape
    kern = _build()
    km = k.mean(dim=0, dtype=torch.float32)        # summed in float32 inside the kernel, no float32 copy of k
    q8, qs = _quantize(q, mult=d ** -0.5 * LOG2E)
    k8, ks = _quantize(k, mean=km)
    o = torch.empty((n, heads, d), dtype=q.dtype, device=q.device)
    kern["attn"][(kern["cdiv"](n, BLOCK_M), heads)](
        q8, k8, qs, ks, v, o, n, v.stride(0), v.stride(1), o.stride(0), o.stride(1),
        PV_F16=pv_f16, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=WARPS, num_stages=STAGES)
    return o[None] if batched else o


def usable(q: torch.Tensor) -> bool:
    """Whether ``attention`` takes this call (the self-check passed once on this process's card)."""
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and q.shape[-1] == HEAD_DIM
        and q.shape[-3] >= MIN_TOKENS
        and q.stride(-1) == 1
        and not torch.is_grad_enabled()
        and bool(_state["ok"])
    )


def self_check(device: torch.device) -> tuple[bool, str]:
    """The kernels once on a random table (1000 tokens: a masked last block; v strided as in the packed qkv),
    against fp32 attention: finite, and within 3 % relative L2 (they measure well under 1 %)."""
    try:
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < (8, 0):
            return False, f"sm_{major}{minor}: Triton's int8 tensor-core products need sm_80"
        g = torch.Generator(device=device).manual_seed(0)
        qkv = torch.randn((1000, 3, 4, HEAD_DIM), device=device, dtype=torch.float16, generator=g)
        q, k, v = qkv.unbind(1)
        want = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(0, 1), k.float().transpose(0, 1), v.float().transpose(0, 1)).transpose(0, 1)
        for pv_f16 in (True, False):
            got = attention(q, k, v, pv_f16=pv_f16).float()
            torch.cuda.synchronize(device)
            rel = float((got - want).norm() / want.norm())
            if not bool(torch.isfinite(got).all()) or not rel <= 0.03:
                return False, f"answer {rel:.3g} from fp32 attention (pv_f16={pv_f16})"
    except Exception as exc:  # noqa: BLE001 - whatever Triton or the card raise: the hook stays unset
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    return True, ""


def state() -> dict:
    """{"ok": None (not tried) / True / False, "why": the reason it is off}."""
    return dict(_state)


def install(pv_f16: bool = True, device: torch.device | None = None) -> bool:
    """Set ``accel.self_attention`` to this kernel after ``self_check`` on ``device`` (default: the current
    card); calls it cannot take (``usable``) decline, and the transformer keeps flash-attn / SDPA for them. A call
    whose kernel fails (anything but the card's out-of-memory or CUDA errors, which propagate) uninstalls the hook
    with a warning and declines."""
    if _state["ok"] is None:
        dev = device or torch.device("cuda", torch.cuda.current_device())
        _state["ok"], _state["why"] = self_check(dev)
    if not _state["ok"]:
        return False

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        if not usable(q):
            return None
        try:
            return attention(q, k, v, pv_f16=pv_f16)
        except Exception as exc:
            if _card_failure(exc):
                raise
            # the kernel's own failure (a Triton compile or launch limit on a new shape): off for good
            _state["ok"], _state["why"] = False, f"{type(exc).__name__}: {str(exc)[:300]}"
            if accel.self_attention is run:
                accel.self_attention = None
            warnings.warn(f"the int8 self-attention failed and is off from now on (flash-attn / SDPA run): "
                          f"{_state['why']}", RuntimeWarning, stacklevel=2)
            return None

    accel.self_attention = run
    return True


def _card_failure(exc: BaseException) -> bool:
    """Whether ``exc`` comes from the card (out of memory, a CUDA error) rather than from the kernel itself."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        return True
    text = str(exc)
    return "CUDA error" in text or "[CUDA]" in text or "out of memory" in text.lower()


def uninstall() -> None:
    """Back to flash-attn / SDPA."""
    accel.self_attention = None
