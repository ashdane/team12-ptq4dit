"""Single-pass attention for DiT-XL/2 tokens (T = 256, head dim 72), read straight from the QKV GEMM.

The whole key/value sequence of one (image, head) fits on chip, so each program computes a block of
queries against all 256 keys with an exact softmax (no online rescaling). The head dimension 72 is
split into a 64-wide and a 16-wide slice (lanes 72..79 masked to zero), so the tensor cores do 80
lanes of work instead of a padded 128. Q, K and V are read from the [M, 3C] QKV output in place
and the result is written in the [M, C] token-major layout the projection consumes, so no permute
or transpose copies are needed.

Precision follows FlashAttention: FP16 inputs, FP32 scores and softmax, FP16 probabilities into the
P @ V product with FP32 accumulation.
"""
import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for bm, w, s in ((32, 4, 1), (32, 4, 2), (64, 4, 1), (64, 4, 2), (64, 8, 1), (64, 8, 2),
                     (128, 8, 1), (128, 8, 2), (16, 4, 1), (32, 2, 1)):
        out.append(triton.Config({"BM": bm}, num_warps=w, num_stages=s))
    return out


@triton.autotune(configs=_configs(), key=["T", "H", "D", "NIMG"])
@triton.jit
def _attn(QKV, O, qk_scale, NIMG, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    C: tl.constexpr = H * D
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid_m * BM + tl.arange(0, BM)
    keys = tl.arange(0, T)
    d0 = tl.arange(0, 64)
    d1 = 64 + tl.arange(0, 16)
    m1 = d1 < D
    tok = b.to(tl.int64) * T
    qrow = QKV + (tok + rows)[:, None] * (3 * C) + h * D
    krow = QKV + (tok + keys)[:, None] * (3 * C) + C + h * D
    vrow = krow + C
    q0 = tl.load(qrow + d0[None, :])
    q1 = tl.load(qrow + d1[None, :], mask=m1[None, :], other=0.0)
    k0 = tl.load(krow + d0[None, :])
    k1 = tl.load(krow + d1[None, :], mask=m1[None, :], other=0.0)
    s = tl.dot(q0, tl.trans(k0)) + tl.dot(q1, tl.trans(k1))          # [BM, T] FP32
    s = s * qk_scale                                                   # includes log2(e)
    mx = tl.max(s, 1)
    p = tl.exp2(s - mx[:, None])
    inv = 1.0 / tl.sum(p, 1)
    p16 = p.to(tl.float16)
    v0 = tl.load(vrow + d0[None, :])
    v1 = tl.load(vrow + d1[None, :], mask=m1[None, :], other=0.0)
    o0 = tl.dot(p16, v0) * inv[:, None]
    o1 = tl.dot(p16, v1) * inv[:, None]
    orow = O + (tok + rows)[:, None] * C + h * D
    tl.store(orow + d0[None, :], o0.to(tl.float16))
    tl.store(orow + d1[None, :], o1.to(tl.float16), mask=m1[None, :])


def attention_impl(qkv, n_img, tokens, heads):
    m, c3 = qkv.shape
    c = c3 // 3
    d = c // heads
    if qkv.dtype != torch.float16 or not qkv.is_contiguous() or m != n_img * tokens:
        raise ValueError("expected contiguous FP16 qkv [n_img * tokens, 3C]")
    if tokens & (tokens - 1) or not 64 < d <= 80:
        raise ValueError("kernel assumes a power-of-two token count and 64 < head dim <= 80")
    o = torch.empty((m, c), device=qkv.device, dtype=torch.float16)
    scale = (d ** -0.5) * 1.4426950408889634
    grid = lambda meta: (tokens // meta["BM"], n_img * heads)
    _attn[grid](qkv, o, scale, n_img, T=tokens, H=heads, D=d)
    return o


@torch.library.custom_op("fc1bench::dit_attention", mutates_args=())
def dit_attention(qkv: torch.Tensor, n_img: int, tokens: int, heads: int) -> torch.Tensor:
    return attention_impl(qkv, n_img, tokens, heads)


@dit_attention.register_fake
def _(qkv, n_img, tokens, heads):
    return qkv.new_empty((qkv.shape[0], qkv.shape[1] // 3))


def attention_ref(qkv, n_img, tokens, heads):
    m, c3 = qkv.shape
    c = c3 // 3
    t = qkv.view(n_img, tokens, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
    o = torch.nn.functional.scaled_dot_product_attention(t[0], t[1], t[2])
    return o.transpose(1, 2).reshape(m, c)
