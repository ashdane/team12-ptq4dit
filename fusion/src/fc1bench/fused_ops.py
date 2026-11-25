"""Fused producer + INT8-quantization kernels and a fused DiT-block forward (v2).

Per DiT block, everything between the INT GEMMs is three single-pass kernels:

  K1 _res_ln_mod_quant : x_new = x + gate*delta (optional residual from the previous branch),
                         LayerNorm (no affine, eps), modulate (x*(1+scale)+shift), then dynamic
                         per-token INT8 quantization. Writes the FP16 residual stream (only when a
                         residual was added), the INT8 activations and the FP32 row scale.
  K2 _act_quant (GELU) : GELU(tanh) of the fc1 output (+ optional channel scale bx), then quant.
  K3 _act_quant (row)  : per-token INT8 quantization of the attention output (proj input).

v2 design (from ncu/nsys profiling on A100 and L4):
  * 2D tiles of ROWS rows x full row width kept in registers; reductions run along the row axis, so
    with several rows per program each row's mean/var/amax reduction stays (mostly) inside a warp
    instead of a cross-warp shared-memory reduction per row (the L4 v1 bottleneck: mio_throttle).
  * Rows split as a power-of-two block + power-of-two tail (1152 = 1024 + 128, 4608 = 4096 + 512):
    no padded lanes.
  * (ROWS, num_warps) autotuned per (M, C): v1 tuned once per C and reused across batch sizes.
  * FAST=True: one IEEE reciprocal per row and y * inv per element instead of y / scale per element,
    and the hardware tanh.approx.f32 for GELU (the A100 v1 GELU kernel was instruction-bound).
    FAST=False reproduces the exact grid of kernels._quantize_fused (IEEE division, libdevice tanh).

Quantization grid: scale = max(rowmax|y| * (1/127), 1e-8), q = clamp(rint(y/scale), -127, 127).
LayerNorm statistics, modulation and GELU run in FP32 (no intermediate FP16 rounding).

The fc2 residual add of block i is deferred into block i+1's K1, and all 28 adaLN projections are
one batched GEMM on SiLU(c) computed once per forward.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice

FAST_DEFAULT = True


def _split(n):
    """n = A + B with A the largest power of two <= n and B the next power of two >= remainder."""
    a = 1 << (n.bit_length() - 1)
    rem = n - a
    b = 1 if rem == 0 else 1 << (rem - 1).bit_length()
    return a, b, rem


def _configs():
    out = []
    for rows, warps in ((1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 4), (4, 8),
                        (8, 4), (8, 8), (16, 8), (4, 2)):
        out.append(triton.Config({"ROWS": rows}, num_warps=warps))
    return out


@triton.jit
def _tanh_fast(x):
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _quant_rows(ya, yb, rows, rmask, mb, ca, cb, Q, S, C: tl.constexpr, FAST: tl.constexpr):
    """ya: [R, A], yb: [R, B] (masked lanes already 0). Writes INT8 rows and FP32 scales."""
    amax = tl.maximum(tl.max(tl.abs(ya), 1), tl.max(tl.abs(yb), 1))
    scale = tl.maximum(amax * (1.0 / 127.0), 1e-8)
    if FAST:
        inv = tl.div_rn(tl.full(scale.shape, 1.0, tl.float32), scale)
        qa = libdevice.rint(ya * inv[:, None])
        qb = libdevice.rint(yb * inv[:, None])
    else:
        qa = libdevice.rint(tl.div_rn(ya, scale[:, None]))
        qb = libdevice.rint(tl.div_rn(yb, scale[:, None]))
    qa = tl.minimum(tl.maximum(qa, -127.0), 127.0).to(tl.int8)
    qb = tl.minimum(tl.maximum(qb, -127.0), 127.0).to(tl.int8)
    base = rows.to(tl.int64)[:, None] * C
    tl.store(Q + base + ca[None, :], qa, mask=rmask[:, None])
    tl.store(Q + base + cb[None, :], qb, mask=rmask[:, None] & mb[None, :])
    tl.store(S + rows, scale, mask=rmask)


@triton.autotune(configs=_configs(), key=["M", "C", "HAS_RES", "FAST"])
@triton.jit
def _res_ln_mod_quant(X, D, G, MOD, XO, Q, S, M, T, g_stride, g_off, mod_stride, shift_off, scale_off, eps,
                      C: tl.constexpr, HAS_RES: tl.constexpr, FAST: tl.constexpr,
                      BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    b = rows // T
    ca = tl.arange(0, BLOCK_A)
    cb = BLOCK_A + tl.arange(0, BLOCK_B)
    mb = cb < C
    ma = rmask[:, None]
    mbb = rmask[:, None] & mb[None, :]
    base = rows.to(tl.int64)[:, None] * C
    xa = tl.load(X + base + ca[None, :], mask=ma, other=0.0).to(tl.float32)
    xb = tl.load(X + base + cb[None, :], mask=mbb, other=0.0).to(tl.float32)
    if HAS_RES:
        gb_ = b[:, None] * g_stride + g_off
        da = tl.load(D + base + ca[None, :], mask=ma, other=0.0).to(tl.float32)
        db = tl.load(D + base + cb[None, :], mask=mbb, other=0.0).to(tl.float32)
        ga = tl.load(G + gb_ + ca[None, :], mask=ma, other=0.0).to(tl.float32)
        gb = tl.load(G + gb_ + cb[None, :], mask=mbb, other=0.0).to(tl.float32)
        xa16 = (xa + ga * da).to(tl.float16)
        xb16 = (xb + gb * db).to(tl.float16)
        tl.store(XO + base + ca[None, :], xa16, mask=ma)
        tl.store(XO + base + cb[None, :], xb16, mask=mbb)
        xa = xa16.to(tl.float32)          # LayerNorm sees exactly the stored residual stream
        xb = xb16.to(tl.float32)
    mean = (tl.sum(xa, 1) + tl.sum(xb, 1)) / C
    xca = xa - mean[:, None]
    xcb = tl.where(mbb, xb - mean[:, None], 0.0)
    var = (tl.sum(xca * xca, 1) + tl.sum(xcb * xcb, 1)) / C
    rstd = tl.div_rn(tl.full(var.shape, 1.0, tl.float32), tl.sqrt_rn(var + eps))
    mo = b[:, None] * mod_stride
    sha = tl.load(MOD + mo + shift_off + ca[None, :], mask=ma, other=0.0).to(tl.float32)
    shb = tl.load(MOD + mo + shift_off + cb[None, :], mask=mbb, other=0.0).to(tl.float32)
    sca = tl.load(MOD + mo + scale_off + ca[None, :], mask=ma, other=0.0).to(tl.float32)
    scb = tl.load(MOD + mo + scale_off + cb[None, :], mask=mbb, other=0.0).to(tl.float32)
    ya = xca * rstd[:, None] * (1.0 + sca) + sha
    yb = tl.where(mbb, xcb * rstd[:, None] * (1.0 + scb) + shb, 0.0)
    _quant_rows(ya, yb, rows, rmask, mb, ca, cb, Q, S, C, FAST)


@triton.autotune(configs=_configs(), key=["M", "C", "HAS_BX", "GELU", "FAST"])
@triton.jit
def _act_quant(H, BX, Q, S, M, C: tl.constexpr, HAS_BX: tl.constexpr, GELU: tl.constexpr, FAST: tl.constexpr,
               BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    ca = tl.arange(0, BLOCK_A)
    cb = BLOCK_A + tl.arange(0, BLOCK_B)
    mb = cb < C
    ma = rmask[:, None]
    mbb = rmask[:, None] & mb[None, :]
    base = rows.to(tl.int64)[:, None] * C
    ha = tl.load(H + base + ca[None, :], mask=ma, other=0.0).to(tl.float32)
    hb = tl.load(H + base + cb[None, :], mask=mbb, other=0.0).to(tl.float32)
    if GELU:
        k0 = 0.7978845608028654       # sqrt(2/pi); GELU(approximate="tanh") as in DiT's Mlp
        ia = k0 * (ha + 0.044715 * ha * ha * ha)
        ib = k0 * (hb + 0.044715 * hb * hb * hb)
        if FAST:
            ha = 0.5 * ha * (1.0 + _tanh_fast(ia))
            hb = 0.5 * hb * (1.0 + _tanh_fast(ib))
        else:
            ha = 0.5 * ha * (1.0 + libdevice.tanh(ia))
            hb = 0.5 * hb * (1.0 + libdevice.tanh(ib))
    if HAS_BX:
        ha = ha * tl.load(BX + ca).to(tl.float32)[None, :]
        hb = hb * tl.load(BX + cb, mask=mb, other=1.0).to(tl.float32)[None, :]
    hb = tl.where(mbb, hb, 0.0)
    ha = tl.where(ma, ha, 0.0)
    _quant_rows(ha, hb, rows, rmask, mb, ca, cb, Q, S, C, FAST)


# --------------------------------------------------------------------------- launchers
def _check(x, name):
    if not x.is_cuda or x.ndim != 2 or x.stride(1) != 1 or x.dtype != torch.float16:
        raise ValueError(f"{name}: expected CUDA FP16 [M, C] with unit last stride, got {x.dtype} {tuple(x.shape)}")


def _grid(m):
    return lambda meta: (triton.cdiv(m, meta["ROWS"]),)


def res_ln_mod_quant_launch(x, mod, shift_off, scale_off, tokens, eps, delta=None, gate_src=None, gate_off=0,
                            fast=FAST_DEFAULT):
    x = x.contiguous()
    _check(x, "x")
    _check(mod, "mod")
    m, c = x.shape
    a, b, _ = _split(c)
    q = torch.empty((m, c), device=x.device, dtype=torch.int8)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    has_res = delta is not None
    xo = torch.empty_like(x) if has_res else x
    if has_res:
        delta = delta.contiguous()
        _check(delta, "delta")
        _check(gate_src, "gate_src")
    g = gate_src if has_res else mod
    _res_ln_mod_quant[_grid(m)](x, delta if has_res else x, g, mod, xo, q, s, m, tokens,
                                g.stride(0), gate_off, mod.stride(0), shift_off, scale_off, eps,
                                C=c, HAS_RES=has_res, FAST=fast, BLOCK_A=a, BLOCK_B=b)
    return xo, q, s


def act_quant_launch(h, gelu, bx=None, fast=FAST_DEFAULT):
    h = h.contiguous()
    _check(h, "h")
    m, c = h.shape
    a, b, _ = _split(c)
    q = torch.empty((m, c), device=h.device, dtype=torch.int8)
    s = torch.empty(m, device=h.device, dtype=torch.float32)
    _act_quant[_grid(m)](h, bx if bx is not None else s, q, s, m, C=c, HAS_BX=bx is not None, GELU=gelu,
                         FAST=fast, BLOCK_A=a, BLOCK_B=b)
    return q, s


# --------------------------------------------------------------------------- custom ops
@torch.library.custom_op("fc1bench::ln_mod_quant", mutates_args=())
def ln_mod_quant(x: torch.Tensor, mod: torch.Tensor, shift_off: int, scale_off: int, tokens: int,
                 eps: float, fast: bool = FAST_DEFAULT) -> Tuple[torch.Tensor, torch.Tensor]:
    _, q, s = res_ln_mod_quant_launch(x, mod, shift_off, scale_off, tokens, eps, fast=fast)
    return q, s


@ln_mod_quant.register_fake
def _(x, mod, shift_off, scale_off, tokens, eps, fast=FAST_DEFAULT):
    return (x.new_empty(x.shape, dtype=torch.int8), x.new_empty((x.shape[0],), dtype=torch.float32))


@torch.library.custom_op("fc1bench::res_ln_mod_quant", mutates_args=())
def res_ln_mod_quant(x: torch.Tensor, delta: torch.Tensor, gate_src: torch.Tensor, gate_off: int,
                     mod: torch.Tensor, shift_off: int, scale_off: int, tokens: int,
                     eps: float, fast: bool = FAST_DEFAULT) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return res_ln_mod_quant_launch(x, mod, shift_off, scale_off, tokens, eps, delta, gate_src, gate_off, fast)


@res_ln_mod_quant.register_fake
def _(x, delta, gate_src, gate_off, mod, shift_off, scale_off, tokens, eps, fast=FAST_DEFAULT):
    return (torch.empty_like(x), x.new_empty(x.shape, dtype=torch.int8),
            x.new_empty((x.shape[0],), dtype=torch.float32))


@torch.library.custom_op("fc1bench::gelu_quant", mutates_args=())
def gelu_quant(h: torch.Tensor, bx: Optional[torch.Tensor] = None,
               fast: bool = FAST_DEFAULT) -> Tuple[torch.Tensor, torch.Tensor]:
    return act_quant_launch(h, True, bx, fast)


@gelu_quant.register_fake
def _(h, bx=None, fast=FAST_DEFAULT):
    return (h.new_empty(h.shape, dtype=torch.int8), h.new_empty((h.shape[0],), dtype=torch.float32))


@torch.library.custom_op("fc1bench::row_quant", mutates_args=())
def row_quant(x: torch.Tensor, fast: bool = FAST_DEFAULT) -> Tuple[torch.Tensor, torch.Tensor]:
    return act_quant_launch(x, False, None, fast)


@row_quant.register_fake
def _(x, fast=FAST_DEFAULT):
    return (x.new_empty(x.shape, dtype=torch.int8), x.new_empty((x.shape[0],), dtype=torch.float32))


# --------------------------------------------------------------------------- references (FP32 torch)
def quantize_ref(y, fast=False):
    scale = (y.abs().amax(dim=1) * (1.0 / 127.0)).clamp_min(1e-8)
    q = torch.round(y * (1.0 / scale)[:, None]) if fast else torch.round(y / scale[:, None])
    return q.clamp(-127, 127).to(torch.int8), scale


def res_ln_mod_quant_ref(x, mod, shift_off, scale_off, tokens, eps, delta=None, gate_src=None, gate_off=0,
                         fast=False):
    m, c = x.shape
    b = torch.arange(m, device=x.device) // tokens
    xf = x.float()
    if delta is not None:
        g = gate_src.float()[b, gate_off:gate_off + c]
        xf = (xf + g * delta.float()).half().float()
    xo = xf.half() if delta is not None else x
    mean = xf.mean(dim=1, keepdim=True)
    xc = xf - mean
    rstd = 1.0 / torch.sqrt((xc * xc).mean(dim=1, keepdim=True) + eps)
    shift = mod.float()[b, shift_off:shift_off + c]
    scale = mod.float()[b, scale_off:scale_off + c]
    y = xc * rstd * (1.0 + scale) + shift
    q, s = quantize_ref(y, fast)
    return xo, q, s


def act_quant_ref(h, gelu, bx=None, fast=False):
    y = h.float()
    if gelu:
        y = F.gelu(y, approximate="tanh")
    if bx is not None:
        y = y * bx.float()
    return quantize_ref(y, fast)


# --------------------------------------------------------------------------- fused DiT forward
def prepare_fused(model):
    """Concatenate all blocks' adaLN Linear weights once: one batched GEMM per forward."""
    if getattr(model, "_fc1bench_adaln_w", None) is None:
        lin = [blk.adaLN_modulation[1] for blk in model.blocks]
        model.register_buffer("_fc1bench_adaln_w", torch.cat([l.weight for l in lin], 0).contiguous(), persistent=False)
        model.register_buffer("_fc1bench_adaln_b", torch.cat([l.bias for l in lin], 0).contiguous(), persistent=False)
        # Re-point every block's adaLN Linear at its slice of the batched buffer instead of keeping
        # a second full FP16 copy of these weights (28 x 6C x C, about 446 MB for DiT-XL/2).
        rows = lin[0].weight.shape[0]
        for i, l in enumerate(lin):
            l.weight = torch.nn.Parameter(model._fc1bench_adaln_w[i * rows:(i + 1) * rows], requires_grad=False)
            l.bias = torch.nn.Parameter(model._fc1bench_adaln_b[i * rows:(i + 1) * rows], requires_grad=False)
    return model


def _int_gemm(layer, q, s, out_dtype=torch.float16):
    from .model_layers import int_gemm_op
    c = layer.config
    return int_gemm_op(q, s, layer.weight, layer.weight_scale, layer.bias, layer.bits, c.bm, c.bn, c.bk,
                       c.warps, c.stages, c.group_m, c.layout, c.backend, layer.out_features,
                       str(out_dtype).split(".")[-1])


def fused_dit_forward(model, x, t, y, block_hook=None, fast=FAST_DEFAULT):
    """DiT.forward with every block running K1 -> qkv -> SDPA -> K3 -> proj -> K1 -> fc1 -> K2 -> fc2.

    Requires a pure-FP16 model whose four projections per block are TunedLinear (INT8/INT4).
    block_hook(i, "start"|"end"), if given, brackets block i (profiling only; not traced by compile).
    """
    prepare_fused(model)
    x = model.x_embedder(x) + model.pos_embed                 # (N, T, C)
    c = model.t_embedder(t) + model.y_embedder(y, False)      # (N, C)
    n_img, tokens, hidden = x.shape
    heads = model.blocks[0].attn.num_heads
    hd = hidden // heads
    m = n_img * tokens
    eps = model.blocks[0].norm1.eps
    six = 6 * hidden
    # All 28 adaLN projections in one GEMM on SiLU(c) (same input for every block).
    mods = F.linear(F.silu(c), model._fc1bench_adaln_w, model._fc1bench_adaln_b)   # (N, 28*6C)
    xs = x.reshape(m, hidden).contiguous()
    pending = None                                            # (fc2 output, mod, gate offset)
    for i, blk in enumerate(model.blocks):
        if block_hook is not None:
            block_hook(i, "start")
        mod = mods[:, i * six:(i + 1) * six]                  # strided view: shift/scale/gate msa, mlp
        if pending is None:
            q, s = ln_mod_quant(xs, mod, 0, hidden, tokens, eps, fast)
        else:
            xs, q, s = res_ln_mod_quant(xs, pending[0], pending[1], pending[2], mod, 0, hidden, tokens, eps, fast)
        qkv = _int_gemm(blk.attn.qkv, q, s).view(n_img, tokens, 3, heads, hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        o = o.transpose(1, 2).reshape(m, hidden)
        q, s = row_quant(o, fast)
        a = _int_gemm(blk.attn.proj, q, s)
        xs, q, s = res_ln_mod_quant(xs, a, mod, 2 * hidden, mod, 3 * hidden, 4 * hidden, tokens, eps, fast)
        h = _int_gemm(blk.mlp.fc1, q, s)
        q, s = gelu_quant(h, None, fast)
        pending = (_int_gemm(blk.mlp.fc2, q, s), mod, 5 * hidden)
        if block_hook is not None:
            block_hook(i, "end")
    mod = pending[1]
    gate = mod[:, 5 * hidden:6 * hidden].repeat_interleave(tokens, dim=0)
    xs = xs + gate * pending[0]
    out = model.final_layer(xs.view(n_img, tokens, hidden), c)
    return model.unpatchify(out)


def fused_dit_forward_v3(model, x, t, y, block_hook=None, fast=FAST_DEFAULT, triton_attn=False):
    """v3 block: LN -> qkv -> SDPA -> row-quant -> proj(+residual) -> LN -> fc1(+GELU+quant) -> fc2(+residual).

    The INT8 GEMMs come from fused_gemm.py: the attention and MLP residual adds run in the proj and
    fc2 epilogues, and fc1 writes INT8 GELU activations with one scale per 128 columns, which fc2
    applies in its main loop. Between GEMMs only LayerNorm+modulate+quant and the attention-output
    quantization remain as separate passes.
    """
    from .fused_gemm import gemm_gelu_gq, gemm_plain, gemm_res
    prepare_fused(model)
    x = model.x_embedder(x) + model.pos_embed
    c = model.t_embedder(t) + model.y_embedder(y, False)
    n_img, tokens, hidden = x.shape
    heads = model.blocks[0].attn.num_heads
    hd = hidden // heads
    m = n_img * tokens
    eps = model.blocks[0].norm1.eps
    six = 6 * hidden
    mods = F.linear(F.silu(c), model._fc1bench_adaln_w, model._fc1bench_adaln_b).contiguous()
    xs = x.reshape(m, hidden).contiguous()
    for i, blk in enumerate(model.blocks):
        if block_hook is not None:
            block_hook(i, "start")
        mod = mods[:, i * six:(i + 1) * six]
        base = i * six
        q, s = ln_mod_quant(xs, mod, 0, hidden, tokens, eps, fast)
        a, p, f1, f2 = blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2
        qkv = gemm_plain(q, s, a.weight, a.weight_scale, a.bias)
        if triton_attn:
            from .fused_attn import dit_attention
            o = dit_attention(qkv, n_img, tokens, heads)
        else:
            qkv = qkv.view(n_img, tokens, 3, heads, hd).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            o = o.transpose(1, 2).reshape(m, hidden)
        q, s = row_quant(o, fast)
        xs = gemm_res(q, s, p.weight, p.weight_scale, p.bias, xs, mods, base + 2 * hidden, tokens, False)
        q, s = ln_mod_quant(xs, mod, 3 * hidden, 4 * hidden, tokens, eps, fast)
        hq, hs = gemm_gelu_gq(q, s, f1.weight, f1.weight_scale, f1.bias)
        xs = gemm_res(hq, hs, f2.weight, f2.weight_scale, f2.bias, xs, mods, base + 5 * hidden, tokens, True)
        if block_hook is not None:
            block_hook(i, "end")
    out = model.final_layer(xs.view(n_img, tokens, hidden), c)
    return model.unpatchify(out)


def fused_dit_forward_v4(model, x, t, y, block_hook=None, fast=FAST_DEFAULT, variant="cg", group=128,
                         tuning=False):
    """v4 block: CUTLASS INT8 GEMMs (cutlass_gemm.py) with fused scale/bias/residual epilogues.

    variant "cg": CUTLASS qkv and proj(+attention residual); fc1(+GELU+group quant) and grouped
                  fc2(+MLP residual) from fused_gemm.py with `group` columns per scale (v3 numerics).
    variant "cc": CUTLASS for all four projections with per-token activation scales (v2 numerics):
                  fc1 -> GELU+quant kernel -> fc2(+MLP residual).
    tuning=True runs eagerly and picks the CUTLASS tile config for every shape it meets.
    """
    from . import cutlass_gemm as cgm
    from .fused_gemm import gemm_gelu_gq, gemm_res
    prepare_fused(model)
    x = model.x_embedder(x) + model.pos_embed
    c = model.t_embedder(t) + model.y_embedder(y, False)
    n_img, tokens, hidden = x.shape
    heads = model.blocks[0].attn.num_heads
    hd = hidden // heads
    m = n_img * tokens
    eps = model.blocks[0].norm1.eps
    six = 6 * hidden
    mods = F.linear(F.silu(c), model._fc1bench_adaln_w, model._fc1bench_adaln_b).contiguous()
    xs = x.reshape(m, hidden).contiguous()

    def plain(q, s, lin):
        if tuning:
            cgm.tune("plain", q, s, lin.weight, lin.weight_scale, lin.bias, tokens)
        return cgm.cutlass_gemm(q, s, lin.weight, lin.weight_scale, lin.bias, tokens)

    def res(q, s, lin, r, off):
        if tuning:
            cgm.tune("res", q, s, lin.weight, lin.weight_scale, lin.bias, tokens, r, mods, off)
        return cgm.cutlass_gemm_res(q, s, lin.weight, lin.weight_scale, lin.bias, r, mods, off, tokens)

    for i, blk in enumerate(model.blocks):
        if block_hook is not None:
            block_hook(i, "start")
        mod = mods[:, i * six:(i + 1) * six]
        base = i * six
        q, s = ln_mod_quant(xs, mod, 0, hidden, tokens, eps, fast)
        a, p, f1, f2 = blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2
        qkv = plain(q, s, a).view(n_img, tokens, 3, heads, hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        o = o.transpose(1, 2).reshape(m, hidden)
        q, s = row_quant(o, fast)
        xs = res(q, s, p, xs, base + 2 * hidden)
        q, s = ln_mod_quant(xs, mod, 3 * hidden, 4 * hidden, tokens, eps, fast)
        if variant == "cg":
            hq, hs = gemm_gelu_gq(q, s, f1.weight, f1.weight_scale, f1.bias, group)
            xs = gemm_res(hq, hs, f2.weight, f2.weight_scale, f2.bias, xs, mods, base + 5 * hidden, tokens, True)
        else:
            hq, hs = gelu_quant(plain(q, s, f1), None, fast)
            xs = res(hq, hs, f2, xs, base + 5 * hidden)
        if block_hook is not None:
            block_hook(i, "end")
    out = model.final_layer(xs.view(n_img, tokens, hidden), c)
    return model.unpatchify(out)


V4_VARIANTS = {"v4": ("cg", 128), "v4w": ("cg", 256), "v4c": ("cc", 128)}


def fused_forward_with_cfg(model, x, t, y, cfg_scale, block_hook=None, fast=FAST_DEFAULT, version=2,
                           triton_attn=False, v4="v4", tuning=False):
    """Same CFG combination as the official DiT.forward_with_cfg (guidance on 3 channels)."""
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    if version == 4:
        variant, group = V4_VARIANTS[v4]
        model_out = fused_dit_forward_v4(model, combined, t, y, block_hook, fast, variant, group, tuning)
    elif version == 3:
        model_out = fused_dit_forward_v3(model, combined, t, y, block_hook, fast, triton_attn)
    else:
        model_out = fused_dit_forward(model, combined, t, y, block_hook, fast)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)
