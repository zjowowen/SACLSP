import torch
from torch import nn
from .distribution import Distribution

class Flow(Distribution):

    def __init__(self, base_dist, transform):
        super().__init__()
        self.base_dist = base_dist
        self.transform = transform

    def forward(self, x, condition=None, **kwargs):
        return self.transform(x, condition, **kwargs)
    
    def log_prob(self, x, condition=None, **kwargs):
        z, log_det = self.transform.inverse(x, condition, **kwargs)
        log_prob_z = self.base_dist.log_prob(z, condition, **kwargs)
        return log_prob_z - log_det
    
    def rsample_and_log_prob(self, condition=None, sample_shape=torch.Size(), **kwargs):
        z, log_prob_z = self.base_dist.rsample_and_log_prob(condition, sample_shape=sample_shape, **kwargs)
        x, log_det = self.transform(z, condition, **kwargs)
        log_prob = log_prob_z + log_det
        return x, log_prob

    def sample_and_log_prob(self, condition=None, sample_shape=torch.Size(), **kwargs):
        with torch.no_grad():
            return self.rsample_and_log_prob(condition, sample_shape=sample_shape, **kwargs)
    
    def rsample(self, condition=None, sample_shape=torch.Size(), **kwargs):
        x, log_prob = self.rsample_and_log_prob(condition, sample_shape=sample_shape, **kwargs)
        return x
    
    def sample(self, condition=None, sample_shape=torch.Size(), **kwargs):
        with torch.no_grad():
            return self.rsample(condition, sample_shape=sample_shape, **kwargs)
