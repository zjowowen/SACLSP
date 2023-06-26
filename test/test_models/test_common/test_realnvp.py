import unittest

import torch
from torch import nn
from torch.nn import functional as F

from SACLSP.models.common import RealNVP

class TestRealNVP(unittest.TestCase):
    def setUp(self):
        self.feature_dim = 10
        self.hidden_dim = 32
        self.num_layers = 2
        self.num_blocks_per_layer = 2

        self.model = RealNVP(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_blocks_per_layer=self.num_blocks_per_layer,
            activation=torch.relu,
            dropout_probability=0,
            use_volume_preserving=False,
            batch_norm_within_layers=False
        )

    def test_model_output_shape(self):
        input_data = torch.randn(5, self.feature_dim)
        z, log_det = self.model.forward(input_data)
        self.assertEqual(z.shape, input_data.shape)
        self.assertEqual(log_det.shape, input_data[:,0].shape)

    def test_invertibility(self):
        input_data = torch.randn(5, self.feature_dim)
        z, log_det = self.model.forward(input_data)
        reconstructed_input, log_det_inverse = self.model.inverse(z)
        self.assertEqual(reconstructed_input.shape, input_data.shape)
        self.assertTrue(torch.allclose(reconstructed_input, input_data))
        self.assertTrue(torch.allclose(log_det, -log_det_inverse))


if __name__ == '__main__':
    
    feature_dim = 10
    hidden_dim = 32
    num_layers = 2
    num_blocks_per_layer = 2

    model = RealNVP(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_blocks_per_layer=num_blocks_per_layer,
        activation=torch.relu,
        dropout_probability=0,
        use_volume_preserving=False,
        batch_norm_within_layers=False
    )

    input_data = torch.randn(5, feature_dim)
    z, log_det = model.forward(input_data)
    reconstructed_input, log_det_inverse = model.inverse(z)
    b=1
