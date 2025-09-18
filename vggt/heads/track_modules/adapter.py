# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                dim_in=None,
                bottleneck=64,
                dim_out=None,
                dropout=0.1,
                #init_option="lora",
                adapter_scalar="0.6",
                adapter_layernorm_option="in"):
        super().__init__()
        self.dim_in = dim_in
        self.down_size = bottleneck
        self.dim_out = dim_out

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm = nn.LayerNorm(self.dim_in)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.dim_in, self.down_size)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.dim_out)

        self.dropout = dropout
    #    if init_option == "bert":
    #        raise NotImplementedError
    #    elif init_option == "lora":
    #        with torch.no_grad():
    #            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
    #            nn.init.zeros_(self.up_proj.weight)
    #            nn.init.zeros_(self.down_proj.bias)
    #            nn.init.zeros_(self.up_proj.bias)

    #def reset_parameters(self):
    #    with torch.no_grad():
    #        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
    #        nn.init.zeros_(self.up_proj.weight)
    #        nn.init.zeros_(self.down_proj.bias)
    #        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm(x)

        down = self.down_proj(x)
        down = self.act(down)
        down = nn.functional.dropout(down, p=self.dropout)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output