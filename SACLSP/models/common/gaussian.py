import torch
from torch import nn
from .mlp import MLP
from .parameter import NonegativeParameter
from .matrix import CovarianceMatrix
from torch.distributions import TransformedDistribution, MultivariateNormal, Independent
from torch.distributions.transforms import TanhTransform
from .distribution import Distribution

from SACLSP.utils.log import log

class StandardGaussian(Distribution):
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.dist=MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    def log_prob(self, x, condition=None, **kwargs):
        return self.dist.log_prob(x)
    
    def rsample_and_log_prob(self, condition=None, sample_shape=torch.Size(), **kwargs):
        if condition is not None:
            sample_shape=condition.shape[0]
        x=self.dist.rsample(sample_shape=sample_shape)
        log_prob=self.dist.log_prob(x)
        return x, log_prob
    
    def sample_and_log_prob(self, condition=None, sample_shape=torch.Size(), **kwargs):
        with torch.no_grad():
            return self.rsample_and_log_prob(condition, sample_shape, **kwargs)
        
    def rsample(self, condition=None, sample_shape=torch.Size(), **kwargs):
        if condition is not None:
            sample_shape=condition.shape[0]
        return self.dist.rsample(sample_shape=sample_shape)

    def sample(self, condition=None, sample_shape=torch.Size(), **kwargs):
        with torch.no_grad():
            return self.rsample(condition=condition, sample_shape=sample_shape, **kwargs)

    def entropy(self):
        return self.dist.entropy()

    def dist(self):
        return self.dist


class Gaussian(Distribution):
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
        return MultivariateNormal(loc=mu, scale_tril = scale_tril)

    def log_prob(self, x, conditioning):
        return self.dist(conditioning).log_prob(x)

    def sample(self, conditioning, sample_shape=torch.Size()):
        return self.dist(conditioning).sample(sample_shape=sample_shape)
            

    def rsample(self, conditioning, sample_shape=torch.Size()): 
        return self.dist(conditioning).rsample(sample_shape=sample_shape)

    def entropy(self, conditioning):
        return self.dist(conditioning).entropy()

    def rsample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        dist=self.dist(conditioning)
        x=dist.rsample(sample_shape=sample_shape)
        log_prob=dist.log_prob(x)
        return x, log_prob

    def sample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(conditioning, sample_shape)

    def forward(self, conditioning):
        dist=self.dist(conditioning)
        x=dist.rsample()
        log_prob=dist.log_prob(x)
        return x, log_prob

class GaussianTanh(Distribution):
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

    def rsample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        dist=self.dist(conditioning)
        x=dist.rsample(sample_shape=sample_shape)
        log_prob=dist.log_prob(x)
        return x, log_prob
    
    def sample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(conditioning, sample_shape)

    def entropy(self, conditioning):
        raise NotImplementedError
        # return self.dist(conditioning).entropy()

    def forward(self, conditioning):
        dist=self.dist(conditioning)
        x=dist.rsample()
        log_prob=dist.log_prob(x)
        return x, log_prob
