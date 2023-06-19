import torch
from torch import nn
from .parameter import NonegativeParameter, TanhParameter
from .function import NonegativeFunction, TanhFunction

class CovarianceMatrix(nn.Module):

    def __init__(self, cfg=None, delta=1e-4):
        super().__init__()
        self.dim = cfg.dim
        if cfg.functional:
            self.functional = True
            self.sigma_lambda = NonegativeFunction(cfg.sigma_lambda)
            self.sigma_offdiag = TanhFunction(cfg.sigma_offdiag)
        else:
            self.functional = False
            if cfg.random_init:
                self.sigma_lambda = NonegativeParameter(torch.abs(nn.init.normal_(torch.Tensor(self.dim))))
                self.sigma_offdiag = TanhParameter(torch.tanh(nn.init.normal_(torch.Tensor(self.dim*(self.dim-1)//2))))
            else:
                self.sigma_lambda = NonegativeParameter(torch.ones(self.dim))
                self.sigma_offdiag = TanhParameter(torch.tanh(torch.zeros(self.dim*(self.dim-1)//2)))
        # register eye matrix
        self.eye = nn.Parameter(torch.eye(self.dim), requires_grad=False)
        self.delta = nn.Parameter(delta*torch.ones(self.dim), requires_grad=False)
        
    def low_triangle_matrix(self, x=None):
        low_t_m = self.eye.clone()
        if self.functional:
            low_t_m[torch.tril_indices(self.dim, self.dim, offset=-1).tolist()]=self.sigma_offdiag(x)
            low_t_m = torch.mul(self.delta+self.sigma_lambda(x),torch.mul(low_t_m, self.delta+self.sigma_lambda(x)).T).T
        else:
            low_t_m[torch.tril_indices(self.dim, self.dim, offset=-1).tolist()]=self.sigma_offdiag.data
            low_t_m = torch.mul(self.delta+self.sigma_lambda.data,torch.mul(low_t_m, self.delta+self.sigma_lambda.data).T).T
        return low_t_m
    
    def forward(self,x=None):
        return torch.matmul(self.low_triangle_matrix(x), self.low_triangle_matrix(x).T)
