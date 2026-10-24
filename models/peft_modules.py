import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import normal

class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
    
    @property
    def dtype(self):
        return self.prompt.dtype

    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    @property
    def dtype(self):
        return self.ln.weight.dtype
    
    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x


class SG_Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, patch_size, dtype=None,
                 Sage_adapter_alpha=0.1, Sage_adapter_use_text_projection=True):
        super().__init__()
        vpt_len = 10
        if not 0.0 <= float(Sage_adapter_alpha) <= 1.0:
            raise ValueError(f"Sage_adapter_alpha must be in [0, 1], got {Sage_adapter_alpha}")
        self.Sage_adapter_alpha = float(Sage_adapter_alpha)
        self.Sage_adapter_use_text_projection = bool(Sage_adapter_use_text_projection)

        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.ln_2 = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)

        self.up_proj = nn.Linear(bottle_dim*2, in_dim, dtype=dtype)
        self.up_proj_2 = nn.Linear(bottle_dim*2, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj_2.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj_2.bias)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        self.down_proj_2 = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        nn.init.kaiming_normal_(self.down_proj_2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj_2.bias)

        self.down_proj_3 = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        nn.init.kaiming_normal_(self.down_proj_3.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj_3.bias)


        self.scale_2 = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.tf = nn.Linear(in_dim, in_dim, dtype=dtype)
        
        self.vpt_len = vpt_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, in_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + in_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)


        self.proj = nn.Linear(197 + vpt_len, 197, dtype=dtype)
        nn.init.kaiming_normal_(self.proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj.bias)

        self.proj_2 = nn.Linear(in_dim, in_dim, dtype=dtype)

    @property
    def dtype(self):
        return self.ln.weight.dtype


    def forward(self, x, text_feat, last_layer):

        text_feat = torch.mean(text_feat, dim=0, keepdim=True).unsqueeze(1).repeat(x.size(0), x.size(1), 1)
        text = self.ln(text_feat)

        x = self.ln(x)
        text_x = (self.proj_2(text) if self.Sage_adapter_use_text_projection else text) * x
        x = self.down_proj(x)

        text_x = self.down_proj_2(text_x)
        text = self.down_proj_3(text)
       
        x = self.Sage_adapter_alpha * self.up_proj(self.relu(torch.cat((x, text_x + self.scale_2 * text), -1)))+\
            (1.0 - self.Sage_adapter_alpha) * self.up_proj(torch.cat((x, text_x + self.scale_2 * text), -1))

        x = x * self.scale

        return x, None
    


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim, dtype=dtype))
        self.scaling = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            return x * self.scale + self.shift

class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, ratio=0.0, generator=None):
        super().__init__()
        # weight: (out_dim, in_dim)
        # bias: (out_dim)
        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        ratio = float(eval(ratio)) if isinstance(ratio, str) else float(ratio)
        num_masked = int(num_params * ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[:num_masked]
        mask = torch.zeros(num_params, dtype=bool).scatter(dim=0, index=masked_indexs, value=True)
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:,:-1]
        self.mask_bias = mask[:,-1]

        self.optimized_weight = nn.Parameter(torch.masked_select(weight.detach(), mask=self.mask_weight))
        self.optimized_bias = nn.Parameter(torch.masked_select(bias.detach(), mask=self.mask_bias))

    def forward(self, x, weight, bias):
        self.mask_weight = self.mask_weight.to(weight.device)
        self.mask_bias = self.mask_bias.to(bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(weight, mask=self.mask_weight, source=self.optimized_weight)
        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(bias, mask=self.mask_bias, source=self.optimized_bias)
        return F.linear(x, weight, bias)