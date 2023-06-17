import torch
from torch import nn
from SACLSP.models.common import MLP
    

class QModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = MLP(cfg.model)
        
    def forward(self, obs, action):
        return self.model(torch.concat((obs,action),dim=1))
    
