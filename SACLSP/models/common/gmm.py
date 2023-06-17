import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .mlp import MLP
from .parameter import NonegativeParameter

class GMM(nn.Module):
    def __init__(self, cfg):
        super(GMM, self).__init__()
        self.num_features = cfg.num_features
        self.num_components = cfg.num_components
        self.conditioning_size = cfg.conditioning_size
        
        # Parameters for the Gaussian Mixture Model
        self.log_w = nn.Parameter(torch.Tensor(cfg.num_components))


        # lower-triangular factor of covariance, with positive-valued diagonal
        self.cov_diag = NonegativeParameter(torch.abs(nn.init.xavier_uniform_(torch.Tensor(cfg.num_components, cfg.num_features))))
        
        # lower-triangular factor of covariance, which is of number num_features*(num_features-1)/2
        self.cov_offdiag = nn.Parameter(torch.Tensor(cfg.num_components, cfg.num_features*(cfg.num_features-1)//2))

        # conbine the diagonal and off-diagonal factors to get the  precision matrix
        
        # nn.Parameter(torch.Tensor(cfg.num_components, cfg.num_features, cfg.num_features))
        
        # Parameters for the conditioning network
        self.conditioning_mu = MLP(cfg.conditioning_mu)
        
    # return the log probability of x given the conditioning 
    def logp(self, x, conditioning):
        pass

    def sample(self, conditioning, num_samples=1):
        pass

    def rsample(self, conditioning, num_samples=1): 
        pass


