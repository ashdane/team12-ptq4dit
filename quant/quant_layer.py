"""
Uniform Affine Quantizer and Quantized Linear Module (`quant/quant_layer.py`)
-----------------------------------------------------------------------------
Implements Eq. (2) of PTQ4DiT:
  Q(x) = clamp(round(x / delta) + lambda, 0, 2^b - 1)
with channel-wise weight quantization and tensor-wise/dynamic activation quantization.

Supports:
  1. Baseline Simulated Quantization (`mode="simulated"`): Reproduces the official
     PTQ4DiT PyTorch fake-quantization behavior (unfused runtime CSB multiply +
     FP16/FP32 round/clamp).
  2. Offline Re-Parameterized Mode (`mode="reparam"`): Completes the unfinished
     TODO in the PTQ4DiT repo by absorbing B_rho^W into W and B_rho^X into
     upstream adaLN MLPs offline, eliminating runtime rescaling overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator (STE) for round-to-nearest."""
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    """
    Uniform Quantizer (Eq. 2 in PTQ4DiT):
      - Channel-wise for weights (`channel_wise=True`)
      - Tensor-wise (or token-wise dynamic) for activations (`channel_wise=False`)
    """

    def __init__(self, n_bits: int = 8, channel_wise: bool = False, dynamic: bool = False):
        super().__init__()
        self.n_bits = int(n_bits)
        self.qmin = 0
        self.qmax = (1 << self.n_bits) - 1
        self.channel_wise = channel_wise
        self.dynamic = dynamic
        self.inited = False

        self.register_buffer("delta", None)
        self.register_buffer("zero_point", None)

    @torch.no_grad()
    def init_quantization_scale(self, x: torch.Tensor):
        if self.channel_wise:
            # Weight tensor shape: (d_out, d_in) -> per output channel scale (d_out, 1)
            x_flat = x.detach().float().view(x.shape[0], -1)
            x_min = x_flat.amin(dim=1, keepdim=True)
            x_max = x_flat.amax(dim=1, keepdim=True)
        else:
            x_min = x.detach().float().amin()
            x_max = x.detach().float().amax()

        x_min = torch.minimum(x_min, torch.zeros_like(x_min))
        x_max = torch.maximum(x_max, torch.zeros_like(x_max))

        delta = (x_max - x_min) / float(self.qmax - self.qmin)
        delta = torch.clamp(delta, min=1e-8)
        zero_point = torch.clamp(torch.round(-x_min / delta), self.qmin, self.qmax)

        self.delta = delta.to(x.dtype)
        self.zero_point = zero_point.to(x.dtype)
        self.inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_bits >= 16:
            return x
        if self.dynamic or not self.inited or self.delta is None:
            self.init_quantization_scale(x)

        delta = self.delta.to(x.dtype)
        zp = self.zero_point.to(x.dtype)
        x_int = round_ste(x / delta) + zp
        x_quant = torch.clamp(x_int, self.qmin, self.qmax)
        x_dequant = (x_quant - zp) * delta
        return x_dequant


class QuantModule(nn.Module):
    """
    Quantized Linear Layer wrapping a DiT `nn.Linear` module with PTQ4DiT
    Channel-wise Salience Balancing (CSB) and Spearman's rho Salience Calibration (SSC).
    """

    def __init__(
        self,
        org_module: nn.Linear,
        weight_bit: int = 8,
        act_bit: int = 8,
        act_dynamic: bool = True,
    ):
        super().__init__()
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight = nn.Parameter(org_module.weight.data.clone())
        if org_module.bias is not None:
            self.bias = nn.Parameter(org_module.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.use_weight_quant = False
        self.use_act_quant = False
        self.reparameterized = False

        self.weight_quantizer = UniformAffineQuantizer(
            n_bits=weight_bit, channel_wise=True, dynamic=False
        )
        self.act_quantizer = UniformAffineQuantizer(
            n_bits=act_bit, channel_wise=False, dynamic=act_dynamic
        )

        # Diagonal vectors of B_rho^X and B_rho^W (shape: [d_in])
        self.register_buffer("bx", torch.ones(self.in_features, dtype=self.weight.dtype, device=self.weight.device))
        self.register_buffer("bw", torch.ones(self.in_features, dtype=self.weight.dtype, device=self.weight.device))

        # Calibration recording buffers
        self.record_salience = False
        self.recorded_act_saliences = []

    @torch.no_grad()
    def record_activation_step(self, x: torch.Tensor):
        # x: (B, N, d_in) or (B, d_in) -> max abs per input channel j
        s_x = x.detach().float().abs().reshape(-1, self.in_features).amax(dim=0)
        self.recorded_act_saliences.append(s_x.cpu())

    @torch.no_grad()
    def apply_csb_ssc(self, use_ssc: bool = True):
        """Computes B_rho^X and B_rho^W from recorded calibration timesteps and initializes weight scale."""
        from .csb_ssc import compute_csb_ssc_balancing_vectors
        if len(self.recorded_act_saliences) == 0:
            act_sal = torch.ones((1, self.in_features), device=self.weight.device)
        else:
            act_sal = torch.stack(self.recorded_act_saliences, dim=0).to(self.weight.device)

        bx, bw, _, _ = compute_csb_ssc_balancing_vectors(act_sal, self.weight.data, use_ssc=use_ssc)
        self.bx.copy_(bx)
        self.bw.copy_(bw)

        # Initialize weight quantizer on balanced weight W_tilde = W * bw (since W is [d_out, d_in])
        w_tilde = self.weight.data * self.bw.unsqueeze(0)
        self.weight_quantizer.init_quantization_scale(w_tilde)
        self.recorded_act_saliences.clear()

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_salience:
            self.record_activation_step(x)

        if not self.reparameterized:
            # Official PTQ4DiT simulated baseline: applies runtime channel balancing
            # X_tilde = X * B_rho^X and W_tilde = B_rho^W * W at every forward pass!
            if self.use_act_quant or self.use_weight_quant:
                x_bal = x * self.bx.to(x.dtype)
                w_bal = self.weight * self.bw.to(self.weight.dtype).unsqueeze(0)
            else:
                x_bal = x
                w_bal = self.weight

            if self.use_weight_quant:
                w_sim = self.weight_quantizer(w_bal)
            else:
                w_sim = w_bal

            if self.use_act_quant:
                x_sim = self.act_quantizer(x_bal)
            else:
                x_sim = x_bal

            return F.linear(x_sim, w_sim, self.bias)
        else:
            # Offline Re-Parameterized Mode (Section 4.3 & Appendix D):
            # B_rho^W and weight quantization are already absorbed into self.weight offline,
            # and B_rho^X is already absorbed into the preceding adaLN / linear layer!
            if self.use_act_quant:
                x_sim = self.act_quantizer(x)
            else:
                x_sim = x
            return F.linear(x_sim, self.weight, self.bias)
