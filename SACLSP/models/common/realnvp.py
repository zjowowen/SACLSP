from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from .gaussian import StandardGaussian
from .flow import Flow
from .transform import AdditiveCouplingTransform, AffineCouplingTransform, CouplingTransform, CompositeTransform
from .resnet import ResidualNet

class RealNVP(Flow):
    """
    RealNVP (Dinh et al., 2016) is a type of normalizing flow that uses an affine coupling layer as a building block.
    https://arxiv.org/abs/1605.08803
    """


    def __init__(self, 
                 feature_dim,
                 hidden_dim,
                 num_layers,
                 num_blocks_per_layer,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_volume_preserving=False,
                 batch_norm_within_layers=False,
                 ):
        
        if use_volume_preserving:
            coupling_constructor=AdditiveCouplingTransform
        else:
            coupling_constructor=AffineCouplingTransform

        mask = torch.ones(feature_dim)
        mask[::2] = -1

        def create_resnet(input_dim, output_dim):
            return ResidualNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                num_blocks=num_blocks_per_layer,
                dropout_probability=dropout_probability,
                batch_norm=batch_norm_within_layers,
            )

        layers=[]
        for i in range(num_layers):
            layers.append(coupling_constructor(transform_net_create_fn=create_resnet,
                                               mask=mask))
            mask = -mask
        
        super().__init__(
            base_dist=StandardGaussian(feature_dim), 
            transform=CompositeTransform(layers)
        )
