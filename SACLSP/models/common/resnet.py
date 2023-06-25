import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        input_dim,
        context_dim,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(input_dim, eps=1e-3) for _ in range(2)]
            )
        if context_dim is not None:
            self.context_layer = nn.Linear(context_dim, input_dim)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, x, context=None):
        x_origin = x
        if self.batch_norm:
            x = self.batch_norm_layers[0](x)
        x = self.activation(x)
        x = self.linear_layers[0](x)
        if self.batch_norm:
            x = self.batch_norm_layers[1](x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_layers[1](x)
        if context is not None:
            x = F.glu(torch.cat((x, self.context_layer(context)), dim=1), dim=1)
        return x + x_origin

class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        context_dim=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        if context_dim is not None:
            self.initial_layer = nn.Linear(
                input_dim + context_dim, hidden_dim
            )
        else:
            self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    input_dim=hidden_dim,
                    context_dim=context_dim,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    batch_norm=batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, context=None):
        if context is None:
            x = self.initial_layer(x)
        else:
            x = self.initial_layer(torch.cat((x, context), dim=1))

        for block in self.blocks:
            x = block(x, context=context)
        outputs = self.final_layer(x)

        return outputs