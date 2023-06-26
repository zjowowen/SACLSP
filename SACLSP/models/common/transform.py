import torch
from torch import nn
from torch.nn.functional import softplus

class Transform(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, condition=None, **kwargs):
        raise NotImplementedError
    
    def inverse(self, x, condition=None, **kwargs):
        raise NotImplementedError
    

class CompositeTransform(Transform):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, condition=None, **kwargs):
        log_det = torch.zeros(x.shape[0])
        for transform in self.transforms:
            x, ld = transform(x, condition, **kwargs)
            log_det += ld
        return x, log_det
    
    def inverse(self, x, condition=None, **kwargs):
        log_det = torch.zeros(x.shape[0])
        for transform in reversed(self.transforms):
            x, ld = transform.inverse(x, condition, **kwargs)
            log_det += ld
        return x, log_det
    

class CouplingTransform(Transform):

    def __init__(self, transform_net_create_fn, mask):
        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)
        self.register_buffer(
            "identity_features", features_vector.masked_select(mask <= 0)
        )
        self.register_buffer(
            "transform_features", features_vector.masked_select(mask > 0)
        )
        assert self.num_identity_features + self.num_transform_features == self.features

        self.mask = mask
        self.transform_net = transform_net_create_fn(
            self.num_identity_features, self.num_transform_features * self._transform_dim_multiplier()
        )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, x, condition=None, **kwargs):
        identity_x = x[:, self.identity_features, ...]
        transform_x = x[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_x, condition, **kwargs)

        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_x, transform_params=transform_params
        )

        output = torch.empty_like(x)
        output[:, self.identity_features, ...] = identity_x
        output[:, self.transform_features, ...] = transform_split
    
        return output, logabsdet
    
    def inverse(self, x, condition=None, **kwargs):
        identity_x = x[:, self.identity_features, ...]
        transform_x = x[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_x, condition, **kwargs)

        transform_split, logabsdet = self._coupling_transform_inverse(
            inputs=transform_x, transform_params=transform_params
        )

        output = torch.empty_like(x)
        output[:, self.identity_features, ...] = identity_x
        output[:, self.transform_features, ...] = transform_split
    
        return output, logabsdet
    
    def log_det(self, transform_params):
        raise NotImplementedError
    
    def _coupling_transform_forward(self, inputs, transform_params):
        raise NotImplementedError
    
    def _coupling_transform_inverse(self, inputs, transform_params):
        raise NotImplementedError
    
    

class AffineCouplingTransform(CouplingTransform):

    DEFAULT_SCALE_ACTIVATION = lambda x : torch.sigmoid(x + 2) + 1e-3
    GENERAL_SCALE_ACTIVATION = lambda x : (softplus(x) + 1e-3).clamp(0, 3)

    def __init__(self, transform_net_create_fn, mask, scale_activation=DEFAULT_SCALE_ACTIVATION):
        self.scale_activation = scale_activation
        super().__init__(transform_net_create_fn, mask)

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        logabsdet = torch.sum(log_scale, dim=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torch.sum(log_scale, dim=1)
        return outputs, logabsdet



class AdditiveCouplingTransform(AffineCouplingTransform):

    def _transform_dim_multiplier(self):
        return 1

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.ones_like(shift)
        return scale, shift

