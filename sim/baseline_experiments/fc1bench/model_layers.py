"""Symmetric quantization ablations on the official pretrained DiT.

These modules do not claim CSB/SSC/AdaRound checkpoint equivalence.
"""
import torch
import torch.nn.functional as F

from .optimized import gemm, prepare_weight, quantize_native
from .quantization import quantize_rows, unpack_int4


class TunedLinear(torch.nn.Module):
    def __init__(self, weight, bias, bits, config, rows, *, fake=False):
        super().__init__()
        q, scale = quantize_rows(weight, bits)
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.bits, self.config, self.rows, self.fake = bits, config, rows, fake
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
        if self.fake:
            return self.reference(x)
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


def convert(model, names, checkpoint, bank, rows, bits, dtype, fake=False):
    records = []
    # checkpoint retains original FP32 weights on CPU; surrounding model can be
    # BF16 without rounding weights before quantization.
    for name in names:
        layer = model.get_submodule(name)
        w = checkpoint[name + ".weight"].to(device="cuda", dtype=torch.float32)
        b = checkpoint.get(name + ".bias")
        b = None if b is None else b.to(device="cuda", dtype=torch.float32)
        config = bank.get(rows, layer.out_features, layer.in_features, bits, dtype)
        replacement = TunedLinear(w, b, bits, config, rows, fake=fake).eval()
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
        from dataclasses import asdict
        records.append(dict(name=name, bits=bits, config=asdict(config),
                            weight_bytes=replacement.weight.numel()*replacement.weight.element_size(),
                            scale_bytes=replacement.weight_scale.numel()*4,
                            bias_bytes=0 if b is None else b.numel()*4))
    return records
