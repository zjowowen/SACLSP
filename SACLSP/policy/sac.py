import torch
from torch import nn
from SACLSP.models.common import GMM, RealNVP, CNF, Gaussian, GaussianTanh


class SACPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.model_type == 'Gaussian':
            self.model = Gaussian(cfg.model)
        elif cfg.model_type == 'GaussianTanh':
            self.model = GaussianTanh(cfg.model)
        elif cfg.model_type == 'GMM':
            self.model = GMM(cfg.model)
        elif cfg.model_type == 'RealNVP':
            self.model = RealNVP(cfg.model)
        elif cfg.model_type == 'CNF':
            self.model = CNF(cfg.model)
        
    def forward(self, obs):
        action, logp = self.model(obs)
        return action, logp
    
    def log_prob(self, action, obs):
        return self.model.log_prob(action, obs)
    
    def sample(self, obs, sample_shape=torch.Size()):
        return self.model.sample(obs, sample_shape)
    
    def rsample(self, obs, sample_shape=torch.Size()):
        return self.model.rsample(obs, sample_shape)
    
    def entropy(self, obs):
        return self.model.entropy(obs)
    
    def dist(self, obs):
        return self.model.dist(obs)
    

if __name__ == "__main__":
    cfg=dict(
            model_type='GaussianTanh',
            model=dict(
                mu_model=dict(
                    hidden_sizes=[11, 64, 32],
                    activation='relu',
                    output_size=3,
                    dropout=0.1,
                    layernorm=True,
                ),
                cov=dict(
                    dim=3,
                ),
            ),
        )
    from easydict import EasyDict
    cfg=EasyDict(cfg)
    sac_policy=SACPolicy(cfg)
    obs=torch.randn(2, 11)
    action, logp = sac_policy(obs)
