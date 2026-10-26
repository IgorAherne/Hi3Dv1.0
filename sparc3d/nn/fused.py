"""Fused fp16 kernels (Triton) for the transformer blocks' elementwise chains, behind ``sparc3d.nn.accel.FUSED``.

Written for Spellcast3D. A block runs a dozen elementwise PyTorch ops per sub-layer, each a full read and write of
the [tokens, 2112] features, many of them in float32: LayerNorm32's two casts, the adaLN modulation, and on q and k
the rotary embedding and the RMS norm. The kernels here do each chain in one read and one write:

    norm_modulate(x, eps, weight, bias, scale1, shift)   LayerNorm32(x) [* weight + bias] [* scale1 + shift]
    rope_rms_(qkv, phases, gamma_q, gamma_k, scale)      in place on the packed qkv [N, 3, H, 96]: q and k
                                                         rotated, then RMS-normalised and scaled

Every value is rounded to fp16 where the eager chain rounds it (by inline PTX: Triton folds a plain round trip) (after LayerNorm32, after the modulation's product
and sum, after the rotation, after the RMS norm), and the rotation multiplies as PyTorch's complex product does, so
only the two reductions (LayerNorm's mean and variance, the norm of q / k) are summed in another order: a change of
rounding, which the samplers amplify into a different sample of the same quality - judged end to end before the
switch is turned on. ``check`` runs both kernels once against the eager ops on a small table; a card or a Triton
that cannot run them turns the switch off with a warning.
"""

from __future__ import annotations

import torch

_kernels: dict | None = None
_state: dict = {"ok": None, "why": ""}
ROWS = 4            # LayerNorm rows per program
TOKENS = 32         # rope_rms tokens per program


def _build() -> dict:
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def r16(x):
        # x rounded to fp16 and back, as the eager op stores it: inline PTX, since Triton folds a plain
        # .to(float16).to(float32) round trip away (measured: the products then match one rounding, not two)
        return tl.inline_asm_elementwise("{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.f32.f16 $0, h; }", "=f,f", [x],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def norm_mod_kernel(X, Y, W, B, S1, SH, n, C, stride_x, stride_y, eps,
                        AFFINE: tl.constexpr, MODULATE: tl.constexpr, BLOCK_C: tl.constexpr,
                        ROWS: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        cmask = cols < C
        if AFFINE:
            w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
        if MODULATE:
            s1 = tl.load(S1 + cols, mask=cmask, other=0.0).to(tl.float32)
            sh = tl.load(SH + cols, mask=cmask, other=0.0).to(tl.float32)
        for r in range(ROWS):
            row = pid * ROWS + r
            if row < n:
                x = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=cmask, other=0.0).to(tl.float32)
                mean = tl.sum(x, axis=0) / C
                xc = tl.where(cmask, x - mean, 0.0)
                var = tl.sum(xc * xc, axis=0) / C
                y = xc * tl.math.rsqrt(var + eps)
                if AFFINE:
                    y = y * w + b
                y = r16(y)
                if MODULATE:
                    y = r16(y * s1) + sh
                tl.store(Y + row.to(tl.int64) * stride_y + cols, y.to(tl.float16), mask=cmask)

    @triton.jit
    def rope_rms_kernel(QKV, PH, GQ, GK, n, stride_n, stride_s, stride_h, scale, eps, TOKENS: tl.constexpr):
        # one head of q (axis 2 = 0) or k (1) for TOKENS tokens: the 96 channels loaded contiguously as 128 lanes,
        # split into the 48 (padded to 64) rotary pairs
        pid = tl.program_id(0)
        h = tl.program_id(1)
        which = tl.program_id(2)
        t = pid * TOKENS + tl.arange(0, TOKENS)
        tmask = t < n
        cols = tl.arange(0, 128)
        m2 = tmask[:, None] & (cols < 96)[None, :]
        base = QKV + t[:, None].to(tl.int64) * stride_n + which * stride_s + h * stride_h + cols[None, :]
        x = tl.load(base, mask=m2, other=0.0).to(tl.float32)
        xe, xo = tl.split(tl.reshape(x, (TOKENS, 64, 2)))
        ph = tl.load(PH + t[:, None].to(tl.int64) * 96 + cols[None, :], mask=m2, other=0.0)
        c, s = tl.split(tl.reshape(ph, (TOKENS, 64, 2)))
        # c10::complex<float> a * b as nvcc contracts it: fma(a.re, b.re, -(a.im * b.im)), fma(a.re, b.im, a.im * b.re)
        re = tl.fma(xe, c, -(xo * s))
        im = tl.fma(xe, s, xo * c)
        re = r16(re)
        im = r16(im)
        nrm = tl.maximum(tl.sqrt_rn(tl.sum(re * re, axis=1) + tl.sum(im * im, axis=1)), eps)
        g = tl.load(tl.where(which == 0, GQ, GK) + h * 96 + cols, mask=cols < 96, other=0.0)
        ge, go = tl.split(tl.reshape(g, (64, 2)))
        ye = (tl.div_rn(re, nrm[:, None]) * ge[None, :]) * scale
        yo = (tl.div_rn(im, nrm[:, None]) * go[None, :]) * scale
        y = tl.reshape(tl.join(ye, yo), (TOKENS, 128))
        tl.store(base, y.to(tl.float16), mask=m2)

    _kernels = {"norm_mod": norm_mod_kernel, "rope_rms": rope_rms_kernel, "cdiv": triton.cdiv,
                "p2": triton.next_power_of_2}
    return _kernels


def norm_modulate(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None,
                  bias: torch.Tensor | None = None, scale1: torch.Tensor | None = None,
                  shift: torch.Tensor | None = None) -> torch.Tensor:
    """fp16 [N, C] -> fp16 [N, C]; ``scale1`` = (1 + scale) and ``shift`` as fp16 rows [C] (the values the eager
    broadcast uses), ``weight`` / ``bias`` float32 [C]."""
    k = _build()
    n, c = x.shape
    y = torch.empty_like(x)
    k["norm_mod"][(k["cdiv"](n, ROWS),)](
        x, y, weight if weight is not None else x, bias if bias is not None else x,
        scale1 if scale1 is not None else x, shift if shift is not None else x, n, c, x.stride(0), y.stride(0),
        float(eps), AFFINE=weight is not None, MODULATE=scale1 is not None, BLOCK_C=k["p2"](c), ROWS=ROWS,
        num_warps=8 if c > 2048 else 4)
    return y


def rope_rms_(qkv: torch.Tensor, phases: torch.Tensor, gamma_q: torch.Tensor, gamma_k: torch.Tensor,
              scale: float, eps: float = 1e-12) -> torch.Tensor:
    """In place: qkv fp16 [N, 3, H, 96] (contiguous), phases complex64 [N, 48], gammas float32 [H, 96]."""
    k = _build()
    n, _, heads, _ = qkv.shape
    ph = torch.view_as_real(phases.reshape(n, 48)).reshape(n, 96)
    k["rope_rms"][(k["cdiv"](n, TOKENS), heads, 2)](
        qkv, ph, gamma_q, gamma_k, n, qkv.stride(0), qkv.stride(1), qkv.stride(2), float(scale), float(eps),
        TOKENS=TOKENS, num_warps=4)
    return qkv


def usable(x: torch.Tensor) -> bool:
    """Whether the fused kernels take this call (fp16 on the card, no autograd, the check passed)."""
    if not (x.is_cuda and x.dtype == torch.float16 and not torch.is_grad_enabled()):
        return False
    if _state["ok"] is None:
        _state["ok"], _state["why"] = check(x.device)
        if not _state["ok"]:
            print(f"[FUSED] the fused kernels are off on this card: {_state['why']}", flush=True)
    return bool(_state["ok"])


def check(device: torch.device) -> tuple[bool, str]:
    """Both kernels once on a random table against the eager ops: within one fp16 rounding step of them."""
    try:
        import torch.nn.functional as F

        g = torch.Generator(device=device).manual_seed(0)
        x = torch.randn((37, 2112), device=device, dtype=torch.float16, generator=g)
        s1 = (1 + 0.1 * torch.randn((2112,), device=device, generator=g)).half()
        sh = (0.1 * torch.randn((2112,), device=device, generator=g)).half()
        want = (F.layer_norm(x.float(), (2112,), eps=1e-6).half() * s1) + sh
        got = norm_modulate(x, 1e-6, scale1=s1, shift=sh)
        if not _close(got, want):
            return False, "norm_modulate differs from the eager ops by more than one rounding"
        qkv = torch.randn((37, 3, 2, 96), device=device, dtype=torch.float16, generator=g)
        ang = torch.rand((37, 48), device=device, generator=g) * 6.28
        phases = torch.polar(torch.ones_like(ang), ang)
        gam = 1 + 0.1 * torch.randn((2, 96), device=device, generator=g)
        q, k, v = qkv.unbind(1)
        rot = []
        for t in (q, k):
            c = torch.view_as_complex(t.float().reshape(37, 2, 48, 2)) * phases[:, None]
            rot.append(torch.view_as_real(c).reshape(37, 2, 96).half())
        want_q = (F.normalize(rot[0].float(), dim=-1) * gam * 96 ** 0.5).half()
        want_k = (F.normalize(rot[1].float(), dim=-1) * gam * 96 ** 0.5).half()
        got = rope_rms_(qkv.clone(), phases, gam, gam, 96 ** 0.5)
        if not (_close(got[:, 0], want_q) and _close(got[:, 1], want_k) and torch.equal(got[:, 2], v)):
            return False, "rope_rms_ differs from the eager ops by more than one rounding"
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - whatever Triton or the card raise: the eager ops serve
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    return True, ""


def _close(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Within two fp16 rounding steps of the eager value everywhere."""
    w = want.float()
    step = torch.where(w == 0, torch.full_like(w, 2.0 ** -24), (w.abs().log2().floor() - 10).exp2())
    return bool(((got.float() - w).abs() <= 2 * step + 1e-6).all())
