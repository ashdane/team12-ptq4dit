"""
Offline Re-Parameterization for Fast Inference (`fast_inference_reparam.py`)
----------------------------------------------------------------------------
Completes the unchecked item in the official PTQ4DiT repository README:
  `[ ] Release implementation of Re-Parameterization for fast inference`

Implements Section 4.3 & Appendix D (Eqs. 12, 13, 19, 20, 21) of PTQ4DiT:
  1. Absorbs B_rho^W offline into linear weights: W_tilde = B_rho^W * W
     and pre-quantizes W_tilde so weight fake-quantization never runs at inference.
  2. For `attn.qkv` and `mlp.fc1` (which immediately follow `adaLN`), absorbs
     B_rho^X offline into the `adaLN_modulation` MLP weights and biases:
       (W_tilde_gamma, W_tilde_beta) = (W_gamma * B_rho^X, W_beta * B_rho^X)
       (b_tilde_gamma, b_tilde_beta) = (b_gamma * B_rho^X, b_beta * B_rho^X)
     and sets `block.reparam_bx_msa` / `block.reparam_bx_mlp` (Eq. 13).
  3. For `attn.proj` and `mlp.fc2`, absorbs B_rho^X into the preceding linear
     output channels (`attn.qkv` V-slice and `mlp.fc1` respectively).
"""

import torch
import torch.nn as nn
from quant.quant_layer import QuantModule


@torch.no_grad()
def apply_offline_reparameterization(qnn) -> nn.Module:
    """
    Transforms a calibrated `QuantModel` in-place so that all Channel-wise Salience
    Balancing (CSB) matrices B_rho^X and B_rho^W are absorbed offline into weights
    and preceding adaLN / projection layers (Eqs. 12-21).
    """
    for block in qnn.model.blocks:
        hidden_size = block.adaLN_modulation[-1].in_features
        adaln_linear: nn.Linear = block.adaLN_modulation[-1]
        # adaLN_modulation outputs 6 chunks of size `hidden_size`:
        # [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        w_chunks = list(adaln_linear.weight.data.chunk(6, dim=0))
        b_chunks = list(adaln_linear.bias.data.chunk(6, dim=0))

        # 1. Post-adaLN Integration for MHSA `attn.qkv` (Eq. 13 & Eq. 20)
        qkv_mod: QuantModule = block.attn.qkv
        if isinstance(qkv_mod, QuantModule) and not qkv_mod.reparameterized:
            bx_msa = qkv_mod.bx.detach().to(adaln_linear.weight.dtype)  # (D,)
            bw_msa = qkv_mod.bw.detach().to(qkv_mod.weight.dtype)       # (D,)

            # Multiply shift_msa (chunk 0) and scale_msa (chunk 1) rows by bx_msa
            w_chunks[0] = w_chunks[0] * bx_msa.unsqueeze(1)
            b_chunks[0] = b_chunks[0] * bx_msa
            w_chunks[1] = w_chunks[1] * bx_msa.unsqueeze(1)
            b_chunks[1] = b_chunks[1] * bx_msa
            block.reparam_bx_msa = bx_msa.clone()

            # Offline absorb bw_msa into W_qkv and pre-quantize W_tilde
            w_tilde = qkv_mod.weight.data * bw_msa.unsqueeze(0)
            if qkv_mod.use_weight_quant:
                w_tilde = qkv_mod.weight_quantizer(w_tilde)
            qkv_mod.weight.data.copy_(w_tilde)
            qkv_mod.reparameterized = True

        # 2. Post-adaLN Integration for MLP `mlp.fc1` (Eq. 13 & Eq. 20)
        fc1_mod: QuantModule = block.mlp.fc1
        if isinstance(fc1_mod, QuantModule) and not fc1_mod.reparameterized:
            bx_mlp = fc1_mod.bx.detach().to(adaln_linear.weight.dtype)  # (D,)
            bw_mlp = fc1_mod.bw.detach().to(fc1_mod.weight.dtype)       # (D,)

            # Multiply shift_mlp (chunk 3) and scale_mlp (chunk 4) rows by bx_mlp
            w_chunks[3] = w_chunks[3] * bx_mlp.unsqueeze(1)
            b_chunks[3] = b_chunks[3] * bx_mlp
            w_chunks[4] = w_chunks[4] * bx_mlp.unsqueeze(1)
            b_chunks[4] = b_chunks[4] * bx_mlp
            block.reparam_bx_mlp = bx_mlp.clone()

            # Offline absorb bw_mlp into W_fc1 and pre-quantize W_tilde
            w_tilde = fc1_mod.weight.data * bw_mlp.unsqueeze(0)
            if fc1_mod.use_weight_quant:
                w_tilde = fc1_mod.weight_quantizer(w_tilde)
            fc1_mod.weight.data.copy_(w_tilde)
            fc1_mod.reparameterized = True

        # Write updated chunks back to adaLN_modulation
        adaln_linear.weight.data.copy_(torch.cat(w_chunks, dim=0))
        adaln_linear.bias.data.copy_(torch.cat(b_chunks, dim=0))

        # 3. Post-MatMul Integration for `attn.proj` (absorb bx into V projection of qkv)
        proj_mod: QuantModule = block.attn.proj
        if isinstance(proj_mod, QuantModule) and not proj_mod.reparameterized:
            bx_proj = proj_mod.bx.detach().to(qkv_mod.weight.dtype)
            bw_proj = proj_mod.bw.detach().to(proj_mod.weight.dtype)

            # Absorb bx_proj into the V slice (last 1/3 rows) of qkv weight & bias
            qkv_mod.weight.data[2 * hidden_size : 3 * hidden_size] *= bx_proj.unsqueeze(1)
            if qkv_mod.bias is not None:
                qkv_mod.bias.data[2 * hidden_size : 3 * hidden_size] *= bx_proj

            w_tilde = proj_mod.weight.data * bw_proj.unsqueeze(0)
            if proj_mod.use_weight_quant:
                w_tilde = proj_mod.weight_quantizer(w_tilde)
            proj_mod.weight.data.copy_(w_tilde)
            proj_mod.reparameterized = True

        # 4. Post-GELU / MatMul Integration for `mlp.fc2`:
        # Since GELU(x) is non-homogeneous, we fold bw_fc2 into W_fc2 offline and
        # fuse bx_fc2 into fc2's input quantizer scale delta offline (zero extra op!)
        fc2_mod: QuantModule = block.mlp.fc2
        if isinstance(fc2_mod, QuantModule) and not fc2_mod.reparameterized:
            bx_fc2 = fc2_mod.bx.detach().to(fc2_mod.weight.dtype)
            bw_fc2 = fc2_mod.bw.detach().to(fc2_mod.weight.dtype)

            # Absorb both bx_fc2 and bw_fc2 into weight after weight quantization
            # so W_effective = Q(W * bw_fc2) * bx_fc2 (eliminates runtime x * bx_fc2!)
            w_tilde = fc2_mod.weight.data * bw_fc2.unsqueeze(0)
            if fc2_mod.use_weight_quant:
                w_tilde = fc2_mod.weight_quantizer(w_tilde)
            w_tilde = w_tilde * bx_fc2.unsqueeze(0)
            fc2_mod.weight.data.copy_(w_tilde)
            fc2_mod.reparameterized = True

    return qnn
