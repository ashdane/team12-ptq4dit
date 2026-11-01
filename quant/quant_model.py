"""
QuantModel Wrapper for Diffusion Transformers (`quant/quant_model.py`)
----------------------------------------------------------------------
Recursively replaces target `nn.Linear` layers inside the 28 `DiTBlock` modules
(`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`) with `QuantModule` instances.
"""

import torch
import torch.nn as nn
from .quant_layer import QuantModule


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        self.model = model
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self._wrap_dit_blocks()

    def _wrap_dit_blocks(self):
        for block in self.model.blocks:
            block.attn.qkv = QuantModule(block.attn.qkv, self.weight_bit, self.act_bit)
            block.attn.proj = QuantModule(block.attn.proj, self.weight_bit, self.act_bit)
            block.mlp.fc1 = QuantModule(block.mlp.fc1, self.weight_bit, self.act_bit)
            block.mlp.fc2 = QuantModule(block.mlp.fc2, self.weight_bit, self.act_bit)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)

    def set_record_salience(self, record: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.record_salience = record
                if record:
                    m.recorded_act_saliences.clear()

    def compute_all_csb_ssc(self, use_ssc: bool = True):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.apply_csb_ssc(use_ssc=use_ssc)

    def forward(self, x, t, y):
        return self.model(x, t, y)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        return self.model.forward_with_cfg(x, t, y, cfg_scale)
