import torch
from torch import nn
from torch.distributions.transforms import TanhTransform

class NonegativeParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True, delta=1e-8):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.log_data = nn.Parameter(torch.log(data + delta), requires_grad=requires_grad)
    
    
    def forward(self):
        return torch.exp(self.log_data)
    
    @property
    def data(self):
        return torch.exp(self.log_data)


class TanhParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.transform = TanhTransform(cache_size=1)

        self.data_inv = nn.Parameter(self.transform.inv(data), requires_grad=requires_grad)

    def forward(self):
        return self.transform(self.data_inv)
    
    @property
    def data(self):
        return self.transform(self.data_inv)



class CovarianceMatrix(nn.Module):

    def __init__(self, dim, delta=1e-4, random_init=True):
        super().__init__()
        self.dim = dim
        if random_init:
            self.sigma_lambda = NonegativeParameter(torch.abs(nn.init.normal_(torch.Tensor(dim))))
            self.sigma_offdiag = TanhParameter(torch.tanh(nn.init.normal_(torch.Tensor(dim*(dim-1)//2))))
        else:
            self.sigma_lambda = NonegativeParameter(torch.ones(dim))
            self.sigma_offdiag = TanhParameter(torch.tanh(torch.zeros(dim*(dim-1)//2)))
        # register eye matrix
        self.eye = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.delta = nn.Parameter(delta*torch.ones(dim), requires_grad=False)
        

    @property
    def low_triangle_matrix(self):
        low_t_m = self.eye.clone()
        low_t_m[torch.tril_indices(self.dim, self.dim, offset=-1).tolist()]=self.sigma_offdiag.data
        low_t_m = torch.mul(self.delta+self.sigma_lambda.data,torch.mul(low_t_m, self.delta+self.sigma_lambda.data).T).T
        return low_t_m
    
    def forward(self):
        return torch.matmul(self.low_triangle_matrix, self.low_triangle_matrix.T)
