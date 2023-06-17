from torch import nn

class RealNVP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, obs):
        pass