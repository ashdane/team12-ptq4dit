"""INT8 GEMMs with fused epilogues for the v3 DiT block (fused_ops.fused_dit_forward_v3).

One Triton kernel, specialised by two compile-time switches:

  EPI = 0  plain:     Y = acc * sx * sw + b                                  (FP16 out)
  EPI = 1  residual:  Y = R + gate[row // T] * (acc * sx * sw + b)           (FP16 out)
  EPI = 2  GELU+quant: h = GELU_tanh(acc * sx * sw + b), then INT8 quantization of h with one
                      scale per (row, GQ consecutive columns). Writes Q [M, N] INT8 and S [M, N/GQ].

  A_GROUPED = False  A has one scale per row: sx [M]
  A_GROUPED = True   A has one scale per (row, GA consecutive K): sx [M, K/GA]. The main loop
                     converts each BK slice of the INT32 product to FP32 and applies its group
                     scale (BK divides GA).

The v3 block uses them as: qkv (EPI 0), proj (EPI 1: attention residual), fc1 (EPI 2), fc2
(A_GROUPED + EPI 1: MLP residual). Compared with v2 this removes the FP16 fc1 output and the
GELU+quant pass that re-read it, and the residual adds no longer need the FP16 branch outputs to
be written and re-read by the next LayerNorm kernel.

Quantization grid of the fc1 output groups: scale = max(amax(|h|) / 127, 1e-8),
q = clamp(rint(h * (1 / scale)), -127, 127) (the fused_ops FAST grid, per 128-column group instead
of per row). Tiles always divide the shapes (DiT-XL/2 dimensions), so no bounds masks are needed.
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice

from .fused_ops import _tanh_fast

GQ = 128     # columns per quantization group of the fc1 output (= K group of the fc2 input)

_CONFIGS = [  # (BM, BN, BK, warps, stages)
    (64, 128, 128, 4, 3), (64, 128, 128, 4, 4), (64, 128, 64, 4, 3), (64, 128, 64, 4, 4),
    (128, 64, 128, 4, 3), (128, 128, 128, 8, 3), (128, 128, 128, 8, 2), (128, 128, 64, 8, 3),
    (128, 128, 64, 4, 4), (64, 256, 128, 8, 3), (128, 256, 64, 8, 3), (256, 128, 64, 8, 3),
    (64, 64, 128, 4, 3), (64, 256, 64, 8, 3), (64, 256, 64, 4, 3), (128, 256, 128, 8, 2),
]


def _configs():
    return [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
            for bm, bn, bk, w, s in _CONFIGS]


def _prune(configs, named_args, **kwargs):
    a = {**named_args, **kwargs}          # constexprs arrive as keyword arguments
    m, n, k = a["M"], a["N"], a["K"]
    ga, qg, epi, grouped = a["GA"], a["QG"], a["EPI"], a["A_GROUPED"]
    keep = []
    for c in configs:
        bm, bn, bk = c.kwargs["BM"], c.kwargs["BN"], c.kwargs["BK"]
        if m % bm or n % bn or k % bk:
            continue
        if grouped and ga % bk:
            continue
        if epi == 2 and bn % qg:
            continue
        keep.append(c)
    return keep


@triton.autotune(configs=_configs(), key=["M", "N", "K", "EPI", "A_GROUPED", "GA", "QG"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemm_v3(X, W, SX, SW, B, Y, R, G, S, T, g_stride, g_off,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             EPI: tl.constexpr, A_GROUPED: tl.constexpr, GA: tl.constexpr, QG: tl.constexpr,
             HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nm: tl.constexpr = M // BM
    nn: tl.constexpr = N // BN
    GROUP: tl.constexpr = 8
    group = pid // (GROUP * nn)
    first_m = group * GROUP
    group_size = tl.minimum(nm - first_m, GROUP)
    pm = first_m + (pid % (GROUP * nn)) % group_size
    pn = (pid % (GROUP * nn)) // group_size
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    xp = X + m[:, None] * K + kk[None, :]
    wp = W + n[None, :] * K + kk[:, None]
    sw = tl.load(SW + n)
    if A_GROUPED:
        KG: tl.constexpr = K // GA
        accf = tl.zeros((BM, BN), tl.float32)
        for grp in range(0, KG):
            part = tl.zeros((BM, BN), tl.int32)
            for _j in tl.static_range(GA // BK):      # INT32 within a group, one FP32 rescale per group
                a = tl.load(xp)
                w = tl.load(wp)
                part = tl.dot(a, w, part, out_dtype=tl.int32)
                xp += BK
                wp += BK
            sg = tl.load(SX + m * KG + grp)
            accf += part.to(tl.float32) * sg[:, None]
        y = accf * sw[None, :]
    else:
        acc = tl.zeros((BM, BN), tl.int32)
        for block in range(0, K // BK):
            a = tl.load(xp)
            w = tl.load(wp)
            acc = tl.dot(a, w, acc, out_dtype=tl.int32)
            xp += BK
            wp += BK
        sx = tl.load(SX + m)
        y = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    if HAS_BIAS:
        y += tl.load(B + n)[None, :]
    out = m[:, None] * N + n[None, :]
    if EPI == 0:
        tl.store(Y + out, y.to(tl.float16))
    elif EPI == 1:
        r = tl.load(R + out).to(tl.float32)
        g = tl.load(G + (m // T)[:, None] * g_stride + g_off + n[None, :]).to(tl.float32)
        tl.store(Y + out, (r + g * y).to(tl.float16))
    else:
        k0 = 0.7978845608028654       # sqrt(2/pi); GELU(approximate="tanh") as in DiT's Mlp
        h = 0.5 * y * (1.0 + _tanh_fast(k0 * (y + 0.044715 * y * y * y)))
        NG: tl.constexpr = BN // QG
        h3 = tl.reshape(h, (BM, NG, QG))
        amax = tl.max(tl.abs(h3), 2)
        scale = tl.maximum(amax * (1.0 / 127.0), 1e-8)
        inv = tl.div_rn(tl.full(scale.shape, 1.0, tl.float32), scale)
        q = libdevice.rint(h3 * inv[:, :, None])
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(Y + out, tl.reshape(q, (BM, BN)))
        gcol = pn * NG + tl.arange(0, NG)
        tl.store(S + m[:, None] * (N // QG) + gcol[None, :], scale)


def _launch(x, sx, w, sw, bias, *, epi, grouped, out, r=None, gate=None, g_off=0, tokens=1, s=None, qg=GQ):
    m, k = x.shape
    n = w.shape[0]
    for t in (x, sx, w, sw):
        if not t.is_cuda or not t.is_contiguous():
            raise ValueError("operands must be contiguous CUDA tensors")
    if x.dtype != torch.int8 or w.dtype != torch.int8 or w.shape[1] != k:
        raise ValueError("expected INT8 x [M, K] and INT8 w [N, K]")
    if m % 64 or n % 64 or k % 128:
        raise ValueError(f"shape {m}x{n}x{k} is not tile-aligned")
    if grouped and (sx.ndim != 2 or sx.shape[0] != m or k % sx.shape[1] or (k // sx.shape[1]) % 64):
        raise ValueError("grouped activation scales must be [M, K/G] with G a multiple of 64")
    ga = k // sx.shape[1] if grouped else GQ
    if not grouped and tuple(sx.shape) != (m,):
        raise ValueError("activation scales must be [M]")
    g = gate if gate is not None else sw
    grid = lambda meta: ((m // meta["BM"]) * (n // meta["BN"]),)
    _gemm_v3[grid](x, w, sx, sw, bias if bias is not None else sw, out,
                   r if r is not None else out, g, s if s is not None else sw,
                   tokens, g.stride(0) if gate is not None else 0, g_off,
                   M=m, N=n, K=k, EPI=epi, A_GROUPED=grouped, GA=ga, QG=qg, HAS_BIAS=bias is not None)
    return out


def gemm_plain_impl(x, sx, w, sw, bias):
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.float16)
    return _launch(x, sx, w, sw, bias, epi=0, grouped=False, out=out)


def gemm_res_impl(x, sx, w, sw, bias, r, gate, g_off, tokens, grouped=False):
    if r.dtype != torch.float16 or not r.is_contiguous() or gate.stride(1) != 1:
        raise ValueError("residual must be contiguous FP16 and gate rows unit-stride")
    out = torch.empty_like(r)
    return _launch(x, sx, w, sw, bias, epi=1, grouped=grouped, out=out, r=r, gate=gate, g_off=g_off,
                   tokens=tokens)


def gemm_gelu_gq_impl(x, sx, w, sw, bias, group=GQ):
    m, n = x.shape[0], w.shape[0]
    q = torch.empty((m, n), device=x.device, dtype=torch.int8)
    s = torch.empty((m, n // group), device=x.device, dtype=torch.float32)
    _launch(x, sx, w, sw, bias, epi=2, grouped=False, out=q, s=s, qg=group)
    return q, s


# --------------------------------------------------------------------------- custom ops
@torch.library.custom_op("fc1bench::gemm_plain", mutates_args=())
def gemm_plain(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
               bias: Optional[torch.Tensor]) -> torch.Tensor:
    return gemm_plain_impl(x, sx, w, sw, bias)


@gemm_plain.register_fake
def _(x, sx, w, sw, bias):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float16)


@torch.library.custom_op("fc1bench::gemm_res", mutates_args=())
def gemm_res(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
             bias: Optional[torch.Tensor], r: torch.Tensor, gate: torch.Tensor, g_off: int,
             tokens: int, grouped: bool) -> torch.Tensor:
    return gemm_res_impl(x, sx, w, sw, bias, r, gate, g_off, tokens, grouped)


@gemm_res.register_fake
def _(x, sx, w, sw, bias, r, gate, g_off, tokens, grouped):
    return torch.empty_like(r)


@torch.library.custom_op("fc1bench::gemm_gelu_gq", mutates_args=())
def gemm_gelu_gq(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
                 bias: Optional[torch.Tensor], group: int = GQ) -> Tuple[torch.Tensor, torch.Tensor]:
    return gemm_gelu_gq_impl(x, sx, w, sw, bias, group)


@gemm_gelu_gq.register_fake
def _(x, sx, w, sw, bias, group=GQ):
    m, n = x.shape[0], w.shape[0]
    return x.new_empty((m, n), dtype=torch.int8), x.new_empty((m, n // group), dtype=torch.float32)


# --------------------------------------------------------------------------- FP32 references
def ref_linear(x, sx, w, sw, bias, grouped=False):
    """Exact INT32 products (float64 matmul) with the same scale placement as the kernel."""
    if grouped:
        m, k = x.shape
        g = k // sx.shape[1]
        xs = x.double().view(m, k // g, g)
        ws = w.double().view(w.shape[0], k // g, g)
        parts = torch.einsum("mgk,ngk->mgn", xs, ws)              # INT32-exact per group
        y = (parts * sx.double()[:, :, None]).sum(1)
    else:
        y = (x.double() @ w.double().T) * sx.double()[:, None]
    y = y * sw.double()[None, :]
    if bias is not None:
        y = y + bias.double()
    return y.float()


def ref_gelu_gq(y, group=GQ):
    h = torch.nn.functional.gelu(y, approximate="tanh")
    m, n = h.shape
    h3 = h.view(m, n // group, group)
    scale = (h3.abs().amax(2) * (1.0 / 127.0)).clamp_min(1e-8)
    q = torch.round(h3 * (1.0 / scale)[:, :, None]).clamp(-127, 127).to(torch.int8)
    return q.view(m, n), scale
