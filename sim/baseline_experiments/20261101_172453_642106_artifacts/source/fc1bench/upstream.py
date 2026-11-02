"""Load the real upstream QuantModule without copying or approximating its code."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import torch


def isolated_module(repo, x, weight, bias, bits, scale_method):
    repo = Path(repo).resolve()
    path = repo / "quant" / "quant_layer.py"
    spec = importlib.util.spec_from_file_location("ptq4dit_isolated_quant_layer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                             device=weight.device, dtype=torch.float32)
    linear.weight.data.copy_(weight)
    if bias is not None:
        linear.bias.data.copy_(bias)
    wparams = dict(n_bits=bits, channel_wise=True, scale_method=scale_method)
    aparams = dict(n_bits=8, symmetric=False, channel_wise=False,
                   scale_method=scale_method, leaf_param=True)
    quant = module.QuantModule(linear, wparams, aparams).eval()
    quant.set_quant_state(True, True)
    # Initialization/calibration is outside the timed region. This is an isolated
    # upstream operator, not a reconstructed CSB/SSC checkpoint.
    quant(x)
    x_dq = quant.act_quantizer(x)
    w_dq = quant.weight_quantizer(weight)
    try:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = dict(repository="https://github.com/adreamwu/PTQ4DiT", revision=revision,
                    dirty=dirty, quant_layer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    weight_quant_params=wparams, activation_quant_params=aparams,
                    calibration="isolated tensor initialization; no CSB/SSC/AdaRound checkpoint")
    state = {}
    for name in ("weight_quantizer", "act_quantizer"):
        q = getattr(quant, name)
        state[name] = {key: (getattr(q, key).detach().cpu() if torch.is_tensor(getattr(q, key))
                             else getattr(q, key)) for key in ("delta", "zero_point")}
    return quant, x_dq, w_dq, metadata, state
