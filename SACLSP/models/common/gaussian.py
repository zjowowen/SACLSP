import torch
from torch import nn
from .mlp import MLP
from .parameter import NonegativeParameter
from .matrix import CovarianceMatrix
from torch.distributions import TransformedDistribution, MultivariateNormal, Independent
from torch.distributions.transforms import TanhTransform

from SACLSP.utils.log import log

class Gaussian(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_features = cfg.num_features
        self.mu_model = MLP(cfg.mu_model)
        self.cov = CovarianceMatrix(cfg.cov)
        
    def dist(self, conditioning):
        mu=self.mu_model(conditioning)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix.repeat(mu.shape[0], 1)
        return Independent(MultivariateNormal(loc=mu, scale_tril = scale_tril),reinterpreted_batch_ndims=1)

    def log_prob(self, x, conditioning):
        return self.dist(conditioning).log_prob(x)

    def sample(self, conditioning, sample_shape=torch.Size()):
        self.dist(conditioning).sample(sample_shape=sample_shape)

    def rsample(self, conditioning, sample_shape=torch.Size()): 
        self.dist(conditioning).rsample(sample_shape=sample_shape)

    def entropy(self, conditioning):
        return self.dist(conditioning).entropy()

    def forward(self, conditioning):
        dist=self.dist(conditioning)
        x=dist.rsample(conditioning)
        log_prob=dist.log_prob(x.detach())
        return x, log_prob

class GaussianTanh(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mu_model = MLP(cfg.mu_model)
        self.cov = CovarianceMatrix(cfg.cov)
        self.functional_cov=cfg.cov.functional

    def dist(self, conditioning):
        mu=self.mu_model(conditioning)
        # repeat the sigma to match the shape of mu
        if self.functional_cov:
            scale_tril = self.cov.low_triangle_matrix(conditioning)
        else:
            scale_tril = self.cov.low_triangle_matrix().unsqueeze(0).repeat(mu.shape[0], 1, 1)
        return TransformedDistribution(
            base_distribution=MultivariateNormal(loc=mu, scale_tril = scale_tril),
            transforms=[TanhTransform(cache_size=1)])

    def log_prob(self, x, conditioning):
        return self.dist(conditioning).log_prob(x)

    def sample(self, conditioning, sample_shape=torch.Size()):
        return self.dist(conditioning).sample(sample_shape=sample_shape)

    def rsample(self, conditioning, sample_shape=torch.Size()): 
        return self.dist(conditioning).rsample(sample_shape=sample_shape)

    def entropy(self, conditioning):
        return self.dist(conditioning).entropy()

    def forward(self, conditioning):
        dist=self.dist(conditioning)
        x=dist.rsample()
        log_prob=dist.log_prob(x)
        return x, log_prob
