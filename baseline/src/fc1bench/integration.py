"""Opt-in experiments for ordinary nn.Linear layers and real-input capture.

This does not convert calibrated PTQ4DiT QuantModule/AdaRound state.
"""
from pathlib import Path
import torch
from .quantization import pack_int4, quantize_rows


class ExperimentalLinear(torch.nn.Module):
    def __init__(self, linear, bits=8):
        super().__init__()
        if type(linear) is not torch.nn.Linear or bits not in (4, 8):
            raise ValueError("Only ordinary nn.Linear and bits=4/8 are supported")
        if not linear.weight.is_cuda:
            raise ValueError("Move the source layer to CUDA before conversion")
        self.bits = bits
        self.in_features, self.out_features = linear.in_features, linear.out_features
        q, s = quantize_rows(linear.weight.detach(), bits)
        self.register_buffer("weight", pack_int4(q) if bits == 4 else q)
        self.register_buffer("weight_scale", s)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().float().clone())

    def forward(self, x):
        from .kernels import integer_linear, quantize_activation
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Use .eval() with torch.inference_mode(); this layer is inference-only")
        if x.shape[-1] != self.in_features:
            raise ValueError("Input feature count does not match layer")
        shape, dtype = x.shape, x.dtype
        q, s = quantize_activation(x.reshape(-1, self.in_features).float().contiguous())
        y = integer_linear(q, self.weight, s, self.weight_scale, self.bias, packed=self.bits == 4)
        return y.reshape(*shape[:-1], self.out_features).to(dtype)


def replace_linears(model, names, bits=8):
    """Replace exactly the named ordinary layers; validate all names before edits."""
    replacements = []
    for name in names:
        if not name:
            raise ValueError("Name a child layer, not the root model")
        original = model.get_submodule(name)
        replacement = ExperimentalLinear(original, bits).eval()
        parent_name, _, leaf = name.rpartition(".")
        replacements.append((model.get_submodule(parent_name) if parent_name else model, leaf, replacement))
    for parent, name, replacement in replacements:
        setattr(parent, name, replacement)
    return model


def capture_next_input(layer, path, metadata=None):
    """Capture the next call's real input and raw weight/bias, then remove hook.

    Call before quantization replacement. Supply checkpoint, timestep, block,
    sampler, seed, CFG and revision in metadata. Capturing is not timed.
    """
    path = Path(path)
    if type(layer) is not torch.nn.Linear:
        raise ValueError("Capture an ordinary nn.Linear before quantization conversion")

    def capture(module, inputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(x=inputs[0].detach().float().cpu(),
                        weight=module.weight.detach().float().cpu(),
                        bias=None if module.bias is None else module.bias.detach().float().cpu(),
                        metadata=metadata or {}), path)
        handle.remove()

    handle = layer.register_forward_pre_hook(capture)
    return handle
