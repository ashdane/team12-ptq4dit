"""Explicitly tuned integer GEMM candidates and execution engine for PTQ4DiT."""
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def _triton_jit(fn):
    if HAS_TRITON:
        return triton.jit(fn)
    return fn

from .kernels import quantize_activation
from .quantization import pack_int4


@dataclass(frozen=True)
class Config:
    bm: int = 32
    bn: int = 64
    bk: int = 64
    warps: int = 4
    stages: int = 2
    group_m: int = 8
    layout: str = "nk"
    backend: str = "triton"


def candidates(bits):
    layouts = ("nk", "kn_packed") if bits == 4 else ("nk",)
    for layout in layouts:
        for bm, bn, bk, warps in ((16, 64, 64, 4), (32, 64, 64, 4),
                                  (32, 128, 64, 4), (64, 64, 64, 4),
                                  (64, 128, 64, 4), (64, 128, 128, 8)):
            for stages in (1, 2, 3):
                yield Config(bm, bn, bk, warps, stages, 8, layout)
    if bits == 8 and hasattr(torch, "_int_mm"):
        # Optional, version-recorded PyTorch internal cuBLAS-backed comparator.
        # Unsupported shapes are rejected by its correctness/compile gate.
        yield Config(backend="torch_int_mm")


@_triton_jit
def _tiled(X, W, SX, SW, B, Y, ZX, W_RED,
           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           PACKED: tl.constexpr, PAIR_N: tl.constexpr, RAW: tl.constexpr,
           BIAS: tl.constexpr, HAS_ZX: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP: tl.constexpr):
    """
    Optimized tiled integer GEMM with INT32 accumulation, L2 cache swizzling,
    fused scale/bias epilogue, optional zero-point correction, and configurable output.
    """
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GROUP * nn)
    first_m = group * GROUP
    group_size = tl.minimum(nm - first_m, GROUP)
    pm = first_m + (pid % (GROUP * nn)) % group_size
    pn = (pid % (GROUP * nn)) // group_size
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)

    # Keep K-loop unrolling bounded; tune pipelining instead of unrolling K.
    for block in range(0, tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        if PAIR_N:
            # kn_packed layout: [K, cdiv(N, 2)]
            pair = pn * (BN // 2) + tl.arange(0, BN // 2)
            byte = tl.load(W + k[:, None] * tl.cdiv(N, 2) + pair[None, :],
                           (k[:, None] < K) & (pair[None, :] < tl.cdiv(N, 2)), 0).to(tl.int32)
            lo = (((byte & 15) ^ 8) - 8).to(tl.int8)
            hi = (((byte >> 4) ^ 8) - 8).to(tl.int8)
            w = tl.interleave(lo, hi)
        elif PACKED:
            # nk layout: [N, cdiv(K, 2)]
            byte = tl.load(W + n[None, :] * tl.cdiv(K, 2) + k[:, None] // 2,
                           (n[None, :] < N) & (k[:, None] // 2 < tl.cdiv(K, 2)), 0).to(tl.int32)
            nibble = (byte >> (4 * (k[:, None] % 2))) & 15
            w = ((nibble ^ 8) - 8).to(tl.int8)
        else:
            # unpacked INT8: [N, K]
            w = tl.load(W + n[None, :] * K + k[:, None],
                        (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)

    if RAW:
        tl.store(Y + m[:, None] * N + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < N))
    else:
        sx = tl.load(SX + m, m < M, 0.0)
        sw = tl.load(SW + n, n < N, 0.0)
        if HAS_ZX:
            zx = tl.load(ZX + m, m < M, 0.0)
            w_red = tl.load(W_RED + n, n < N, 0.0)
            acc_corr = acc.to(tl.float32) - zx[:, None] * w_red[None, :]
            out = acc_corr * sx[:, None] * sw[None, :]
        else:
            out = acc.to(tl.float32) * sx[:, None] * sw[None, :]

        if BIAS:
            out += tl.load(B + n, n < N, 0.0)[None, :]
        tl.store(Y + m[:, None] * N + n[None, :], out,
                 (m[:, None] < M) & (n[None, :] < N))


@_triton_jit
def _epilogue(A, SX, SW, B, Y, ZX, W_RED,
              M: tl.constexpr, N: tl.constexpr,
              BIAS: tl.constexpr, HAS_ZX: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < M * N
    row = i // N
    col = i % N
    a = tl.load(A + i, valid, 0).to(tl.float32)
    sx = tl.load(SX + row, valid, 0.0)
    sw = tl.load(SW + col, valid, 0.0)
    if HAS_ZX:
        zx = tl.load(ZX + row, valid, 0.0)
        w_red = tl.load(W_RED + col, valid, 0.0)
        a = a - zx * w_red
    y = a * sx * sw
    if BIAS:
        y += tl.load(B + col, valid, 0.0)
    tl.store(Y + i, y, valid)


def prepare_weight(q, bits, config):
    """
    Packs and organizes weights according to configuration layout.
    q: [N, K] int8 tensor
    """
    if bits == 8:
        if config.layout != "nk":
            raise ValueError("INT8 layout must be nk")
        return q.contiguous()
    if bits != 4 or config.backend != "triton":
        raise ValueError("Packed INT4 requires the Triton backend")
    if config.layout == "kn_packed":
        return pack_int4(q.T.contiguous())
    if config.layout == "nk":
        return pack_int4(q)
    raise ValueError(f"Unknown layout: {config.layout}")


def quantize_native(x, bx=None, affine=False):
    """Fused activation quantization via Triton kernels."""
    return quantize_activation(x, bx=bx, affine=affine)


def gemm(x, weight, sx, sw, bias=None, *, bits=8, config=Config(),
         dtype=torch.float32, zx=None, w_red=None, raw=False, return_kernel=False):
    """
    Tiled integer GEMM with fused scaling, optional zero-point correction, and configurable output.
    Operates strictly on signed INT8 activations and weights (or packed INT4 nibbles).
    Weights are strictly symmetric around zero (zw = 0).

    Zero-point correction:
        corrected_acc = raw_acc - zx * sum(Wq, dim=K)
        out = corrected_acc * sx[:, None] * sw[None, :] + (bias if bias is not None else 0)
    """
    m, k = x.shape
    n = sw.numel()
    if min(m, k, n) < 1 or k > 131071:
        raise ValueError("Invalid/unsafe matrix dimensions for INT32 accumulation (K must be <= 131071)")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported output dtype")
    if x.dtype != torch.int8 or bits not in (4, 8):
        raise TypeError(f"Expected signed int8 activations and bits=4/8, got dtype {x.dtype}")
    expected = (k, (n + 1) // 2) if config.layout == "kn_packed" else (n, (k + 1) // 2 if bits == 4 else k)
    if weight.shape != expected or weight.dtype != (torch.int8 if bits == 8 else torch.uint8):
        raise ValueError(f"Weight shape {weight.shape} / dtype {weight.dtype} disagrees with layout {config.layout}")
    if sx.shape != (m,) or sw.shape != (n,) or sx.dtype != torch.float32 or sw.dtype != torch.float32:
        raise ValueError("Expected FP32 per-row/per-output scales")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32):
        raise ValueError("Bias must be FP32 [N]")

    has_zx = zx is not None
    if has_zx:
        if zx.shape != (m,) or zx.dtype not in (torch.int32, torch.float32):
            raise ValueError("Expected [M] int32/float32 activation zero-points zx")
        if w_red is None:
            # Precompute row reduction sum_k W[n, k] if not provided
            if bits == 4:
                from .quantization import unpack_int4, unpack_int4_kn
                w_unpacked = unpack_int4_kn(weight, n, k) if config.layout == "kn_packed" else unpack_int4(weight, k)
                w_red = w_unpacked.float().sum(dim=1)
            else:
                w_red = weight.float().sum(dim=1)
        if w_red.shape != (n,) or w_red.dtype != torch.float32:
            raise ValueError("Expected [N] float32 weight reduction buffer w_red")

    check_tensors = [x, weight, sx, sw] + ([] if bias is None else [bias]) + ([] if zx is None else [zx])
    for t in check_tensors:
        if not t.is_cuda or t.device != x.device or not t.is_contiguous():
            raise ValueError("All operands must be contiguous on the same GPU")

    y = torch.empty((m, n), device=x.device, dtype=torch.int32 if raw else dtype)
    kernel = None

    if config.backend == "torch_int_mm":
        if bits != 8 or config.layout != "nk":
            raise ValueError("torch_int_mm only supports unpacked INT8")
        with torch.autocast("cuda", enabled=False):
            acc = torch._int_mm(x, weight.T)
        if raw:
            return (acc, None) if return_kernel else acc
        kernel = _epilogue[(triton.cdiv(m * n, 512),)](
            acc, sx, sw, bias if bias is not None else sw, y,
            zx if has_zx else sx, w_red if has_zx else sw,
            m, n, bias is not None, has_zx, 512
        )
    elif config.backend == "triton":
        if config.layout not in ("nk", "kn_packed") or (bits == 8 and config.layout != "nk"):
            raise ValueError(f"Unsupported layout {config.layout} for bits={bits}")
        grid = (triton.cdiv(m, config.bm) * triton.cdiv(n, config.bn),)
        kernel = _tiled[grid](
            x, weight, sx, sw, bias if bias is not None else sw, y,
            zx if has_zx else sx, w_red if has_zx else sw,
            m, n, k, bits == 4, config.layout == "kn_packed", raw,
            bias is not None, has_zx,
            config.bm, config.bn, config.bk, config.group_m,
            num_warps=config.warps, num_stages=config.stages
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    return (y, kernel) if return_kernel else y
