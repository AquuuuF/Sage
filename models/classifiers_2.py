import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        # Paper Eq. (23): learnable s1 and s2, distinct from Sage_lambda_1/2.
        self.scale_1 = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.scale_2 = nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()


class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineClassifier_Sage(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x, f_zs=False, w_zs=False, init_text=None, feat=None):
        if (f_zs==False) and (w_zs==False):
            x = F.normalize(x, dim=-1)
            weight = F.normalize(self.weight, dim=-1)
            output = F.linear(x, weight) * self.scale
        
        elif (f_zs==True) and (w_zs==False):
            x = F.normalize(x, dim=-1)
            weight = F.normalize(self.weight, dim=-1)
            output = F.linear(x, weight) * self.scale * self.scale_1


        elif (f_zs==False) and (w_zs==True):
            x = F.normalize(feat, dim=-1)
            weight = F.normalize(init_text, dim=-1)
            output = F.linear(x, weight) * self.scale * self.scale_2

        return output


class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
    
    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)