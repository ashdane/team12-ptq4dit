"""Explicitly tuned integer GEMM candidates; no hidden tuning in forward()."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .kernels import _quantize
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


@triton.jit
def _tiled(X, W, SX, SW, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           PACKED: tl.constexpr, PAIR_N: tl.constexpr, RAW: tl.constexpr,
           BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP: tl.constexpr):
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
    for block in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
        k = block * BK + kk
        a = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        if PAIR_N:
            pair = pn * (BN // 2) + tl.arange(0, BN // 2)
            byte = tl.load(W + k[:, None] * tl.cdiv(N, 2) + pair[None, :],
                           (k[:, None] < K) & (pair[None, :] < tl.cdiv(N, 2)), 0).to(tl.int32)
            lo = (((byte & 15) ^ 8) - 8).to(tl.int8)
            hi = (((byte >> 4) ^ 8) - 8).to(tl.int8)
            w = tl.interleave(lo, hi)
        elif PACKED:
            byte = tl.load(W + n[None, :] * tl.cdiv(K, 2) + k[:, None] // 2,
                           (n[None, :] < N) & (k[:, None] < K), 0).to(tl.int32)
            nibble = (byte >> (4 * (k[:, None] % 2))) & 15
            w = ((nibble ^ 8) - 8).to(tl.int8)
        else:
            w = tl.load(W + n[None, :] * K + k[:, None],
                        (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    if RAW:
        out = acc
    else:
        sx = tl.load(SX + m, m < M, 0)
        sw = tl.load(SW + n, n < N, 0)
        out = acc.to(tl.float32) * sx[:, None] * sw[None, :]
        if BIAS:
            out += tl.load(B + n, n < N, 0)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], out,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _epilogue(A, SX, SW, B, Y, M: tl.constexpr, N: tl.constexpr,
              BIAS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < M * N
    a = tl.load(A + i, valid, 0).to(tl.float32)
    sx = tl.load(SX + i // N, valid, 0)
    sw = tl.load(SW + i % N, valid, 0)
    y = a * sx * sw
    if BIAS:
        y += tl.load(B + i % N, valid, 0)
    tl.store(Y + i, y, valid)


def prepare_weight(q, bits, config):
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


def quantize_native(x):
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Expected contiguous CUDA [M,K]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Activation dtype must be FP32/FP16/BF16")
    m, k = x.shape
    if not m or not 0 < k <= 16384:
        raise ValueError("Expected M>0 and 1<=K<=16384")
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty(m, device=x.device, dtype=torch.float32)
    _quantize[(m,)](x, q, scale, k, triton.next_power_of_2(k))
    return q, scale


def gemm(x, weight, sx, sw, bias=None, *, bits=8, config=Config(),
         dtype=torch.float32, raw=False, return_kernel=False):
    m, k = x.shape
    n = sw.numel()
    if min(m, k, n) < 1 or k > 131071:
        raise ValueError("Invalid/unsafe matrix dimensions for INT32 accumulation")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported output dtype")
    if x.dtype != torch.int8 or bits not in (4, 8):
        raise ValueError("Expected INT8 activations and bits=4/8")
    expected = (k, (n + 1) // 2) if config.layout == "kn_packed" else (n, (k + 1) // 2 if bits == 4 else k)
    if weight.shape != expected or weight.dtype != (torch.int8 if bits == 8 else torch.uint8):
        raise ValueError("Weight dtype/shape disagrees with layout")
    if sx.shape != (m,) or sw.shape != (n,) or sx.dtype != torch.float32 or sw.dtype != torch.float32:
        raise ValueError("Expected FP32 per-row/per-output scales")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32):
        raise ValueError("Bias must be FP32 [N]")
    for t in [x, weight, sx, sw] + ([] if bias is None else [bias]):
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
        kernel = _epilogue[(triton.cdiv(m*n, 512),)](
            acc, sx, sw, bias if bias is not None else sw, y, m, n, bias is not None, 512,
            enable_fp_fusion=False)
    elif config.backend == "triton":
        if config.layout not in ("nk", "kn_packed") or (bits == 8 and config.layout != "nk"):
            raise ValueError("Unsupported layout")
        grid = (triton.cdiv(m, config.bm) * triton.cdiv(n, config.bn),)
        kernel = _tiled[grid](x, weight, sx, sw, bias if bias is not None else sw, y,
                              m, n, k, bits == 4, config.layout == "kn_packed", raw,
                              bias is not None, config.bm, config.bn, config.bk, config.group_m,
                              num_warps=config.warps, num_stages=config.stages, enable_fp_fusion=False)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
    return (y, kernel) if return_kernel else y
