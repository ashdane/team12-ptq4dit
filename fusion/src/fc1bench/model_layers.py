"""Symmetric quantization ablations on the official pretrained DiT.

These modules do not claim CSB/SSC/AdaRound checkpoint equivalence.
"""
import torch
import torch.nn.functional as F

from typing import Optional

from .optimized import Config, gemm, prepare_weight, quantize_native
from .quantization import quantize_rows, unpack_int4


def _quant_linear_impl(x, weight, weight_scale, bias, bits, bm, bn, bk, warps, stages,
                       group_m, layout, backend, out_features):
    config = Config(bm, bn, bk, warps, stages, group_m, layout, backend)
    q, scale = quantize_native(x)
    return gemm(q, weight, scale, weight_scale, bias, bits=bits, config=config, dtype=x.dtype)


if hasattr(torch, "library") and hasattr(torch.library, "custom_op"):
    # Opaque operator so torch.compile traces the whole model without graph breaks and
    # fuses the surrounding float ops, while our Triton quantize + GEMM run unchanged.
    @torch.library.custom_op("fc1bench::quant_linear", mutates_args=())
    def quant_linear_op(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                        bias: Optional[torch.Tensor], bits: int, bm: int, bn: int, bk: int,
                        warps: int, stages: int, group_m: int, layout: str, backend: str,
                        out_features: int) -> torch.Tensor:
        return _quant_linear_impl(x, weight, weight_scale, bias, bits, bm, bn, bk, warps, stages,
                                  group_m, layout, backend, out_features)

    @quant_linear_op.register_fake
    def _(x, weight, weight_scale, bias, bits, bm, bn, bk, warps, stages, group_m, layout,
          backend, out_features):
        return x.new_empty((x.shape[0], out_features))
else:  # pragma: no cover - old PyTorch
    quant_linear_op = None


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def quantize_torch(x):
    """Dynamic per-token symmetric INT8 quantization written with standard PyTorch ops.

    Same grid as kernels._quantize_fused / quantization.quantize_rows: FP32 math,
    scale = max(rowmax|x| * (1/127), 1e-8), round-half-to-even, clamp to [-127, 127].
    Because these are ordinary ops, torch.compile can fuse them into the kernel that
    produces x (LayerNorm + modulate, GELU, attention-output reshape), which then writes
    INT8 + scale directly instead of an FP16 activation that a separate kernel re-reads.
    """
    xf = x.float()
    scale = (xf.abs().amax(dim=1) * (1.0 / 127.0)).clamp_min(1e-8)
    q = torch.round(xf / scale[:, None]).clamp(-127.0, 127.0).to(torch.int8)
    return q, scale


def _int_gemm_impl(xq, sx, weight, weight_scale, bias, bits, bm, bn, bk, warps, stages,
                   group_m, layout, backend, out_features, out_dtype):
    config = Config(bm, bn, bk, warps, stages, group_m, layout, backend)
    return gemm(xq.contiguous(), weight, sx.contiguous(), weight_scale, bias, bits=bits,
                config=config, dtype=_DTYPES[out_dtype])


if quant_linear_op is not None:
    # GEMM-only opaque op: activation quantization stays visible to the compiler.
    @torch.library.custom_op("fc1bench::int_gemm", mutates_args=())
    def int_gemm_op(xq: torch.Tensor, sx: torch.Tensor, weight: torch.Tensor,
                    weight_scale: torch.Tensor, bias: Optional[torch.Tensor], bits: int, bm: int,
                    bn: int, bk: int, warps: int, stages: int, group_m: int, layout: str,
                    backend: str, out_features: int, out_dtype: str) -> torch.Tensor:
        return _int_gemm_impl(xq, sx, weight, weight_scale, bias, bits, bm, bn, bk, warps, stages,
                              group_m, layout, backend, out_features, out_dtype)

    @int_gemm_op.register_fake
    def _(xq, sx, weight, weight_scale, bias, bits, bm, bn, bk, warps, stages, group_m, layout,
          backend, out_features, out_dtype):
        return xq.new_empty((xq.shape[0], out_features), dtype=_DTYPES[out_dtype])
else:  # pragma: no cover
    int_gemm_op = None


class TunedLinear(torch.nn.Module):
    def __init__(self, weight, bias, bits, config, rows, *, fake=False, io_dtype=None):
        super().__init__()
        q, scale = quantize_rows(weight, bits)
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.bits, self.config, self.rows, self.fake = bits, config, rows, fake
        # None keeps the incoming activation dtype (under autocast this is often FP32,
        # because LayerNorm/GELU emit FP32). A dtype casts input and output to it.
        self.io_dtype = io_dtype
        # Route through the registered custom op (needed for graph-break-free torch.compile).
        # Same kernels and math as the direct path; set by the caller when compiling.
        self.use_custom_op = False
        # "kernel": our fused Triton quantizer (inside the op). "torch": quantize_torch
        # outside a GEMM-only op, so torch.compile can fuse it into the producer kernel.
        self.quant_mode = "kernel"
        self.register_buffer("weight", prepare_weight(q, bits, config))
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", None if bias is None else bias.float().clone())

    def unpack(self):
        if self.bits == 8:
            return self.weight
        if self.config.layout == "kn_packed":
            return unpack_int4(self.weight, self.out_features).T.contiguous()
        return unpack_int4(self.weight, self.in_features)

    def reference(self, x):
        """Identical quantization grid, explicit FP32 QDQ + F.linear."""
        flat = x.reshape(-1, self.in_features)
        q, s = quantize_rows(flat)
        w = self.unpack().float() * self.weight_scale[:, None]
        with torch.autocast("cuda", enabled=False):
            y = F.linear(q.float() * s[:, None], w, self.bias)
        return y.to(x.dtype).reshape(*x.shape[:-1], self.out_features)

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("TunedLinear requires eval() and inference_mode()/no_grad()")
        if x.shape[-1] != self.in_features or x.numel() // self.in_features != self.rows:
            raise ValueError("Input shape differs from the validated tuning shape")
        if self.io_dtype is not None and x.dtype != self.io_dtype:
            x = x.to(self.io_dtype)
        if self.fake:
            return self.reference(x)
        if self.quant_mode == "torch":
            c = self.config
            q, scale = quantize_torch(x.reshape(-1, self.in_features))
            y = int_gemm_op(q, scale, self.weight, self.weight_scale, self.bias, self.bits, c.bm,
                            c.bn, c.bk, c.warps, c.stages, c.group_m, c.layout, c.backend,
                            self.out_features, str(x.dtype).split(".")[-1])
            return y.reshape(*x.shape[:-1], self.out_features)
        if self.use_custom_op:
            c = self.config
            y = quant_linear_op(x.reshape(-1, self.in_features).contiguous(), self.weight,
                                self.weight_scale, self.bias, self.bits, c.bm, c.bn, c.bk, c.warps,
                                c.stages, c.group_m, c.layout, c.backend, self.out_features)
            return y.reshape(*x.shape[:-1], self.out_features)
        q, scale = quantize_native(x.reshape(-1, self.in_features).contiguous())
        y = gemm(q, self.weight, scale, self.weight_scale, self.bias,
                 bits=self.bits, config=self.config, dtype=x.dtype)
        return y.reshape(*x.shape[:-1], self.out_features)


def projection_names(model, scope):
    suffixes = ("mlp.fc1", "mlp.fc2")
    if scope == "all":
        suffixes += ("attn.qkv", "attn.proj")
    elif scope != "mlp":
        raise ValueError("scope must be mlp or all")
    names = [f"blocks.{i}.{suffix}" for i in range(len(model.blocks)) for suffix in suffixes]
    for name in names:
        if type(model.get_submodule(name)) is not torch.nn.Linear:
            raise TypeError(f"{name} is not an ordinary Linear; calibrated models need their own converter")
    return names


def convert(model, names, checkpoint, bank, rows, bits, dtype, fake=False, io_dtype=None):
    records = []
    # checkpoint retains original FP32 weights on CPU; surrounding model can be
    # BF16 without rounding weights before quantization.
    for name in names:
        layer = model.get_submodule(name)
        w = checkpoint[name + ".weight"].to(device="cuda", dtype=torch.float32)
        b = checkpoint.get(name + ".bias")
        b = None if b is None else b.to(device="cuda", dtype=torch.float32)
        config = bank.get(rows, layer.out_features, layer.in_features, bits, dtype)
        replacement = TunedLinear(w, b, bits, config, rows, fake=fake, io_dtype=io_dtype).eval()
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
        from dataclasses import asdict
        records.append(dict(name=name, bits=bits, config=asdict(config),
                            weight_bytes=replacement.weight.numel()*replacement.weight.element_size(),
                            scale_bytes=replacement.weight_scale.numel()*4,
                            bias_bytes=0 if b is None else b.numel()*4))
    return records
