import torch
from torch import nn
from torch.distributions.transforms import TanhTransform
from SACLSP.models.common import MLP

class NonegativeFunction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = MLP(cfg.model)

    def forward(self, x):
        return torch.exp(self.model(x))

class TanhFunction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.transform = TanhTransform(cache_size=1)
        self.model = MLP(cfg.model)

    def forward(self, x):
        return self.transform(self.model(x))
    
