import torch
from torch import nn
from SACLSP.models.common import MLP
    

class QModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_num=cfg.model_num if hasattr(cfg,'model_num') else 1
        self.models=nn.ModuleList([MLP(cfg.model) for _ in range(self.model_num)])
        
    def forward(self, obs, action):
        return torch.concat([model(torch.concat((obs,action),dim=1)) for model in self.models],dim=1)

    def min_q(self, obs, action):
        return torch.min(input=torch.concat([model(torch.concat((obs,action),dim=1)) for model in self.models],dim=1),dim=1).values
