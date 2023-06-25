import unittest

import torch
from torch import nn
from torch.nn import functional as F

from SACLSP.models.common import Transform, CompositeTransform, CouplingTransform, AffineCouplingTransform, AdditiveCouplingTransform, ResidualNet


class TestTransform(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    def test_CompositeTransform(self):


        class TestTransform1(Transform):
            def __init__(self):
                super().__init__()
            def forward(self, x, condition=None, **kwargs):
                return x, torch.zeros_like(x)
            def inverse(self, x, condition=None, **kwargs):
                return x, torch.zeros_like(x)
        
        class TestTransform2(Transform):
            def __init__(self):
                super().__init__()
            def forward(self, x, condition=None, **kwargs):
                return x, torch.ones_like(x)
            def inverse(self, x, condition=None, **kwargs):
                return x, torch.ones_like(x)

        transform = CompositeTransform([TestTransform1(), TestTransform2()])
        x = torch.randn(32, 2)
        y, log_det = transform(x)
        self.assertTrue(torch.allclose(x, y))
        self.assertTrue(torch.allclose(log_det, torch.ones_like(x)+torch.zeros_like(x)))

        transform = CompositeTransform([TestTransform2(), TestTransform1()])
        x = torch.randn(32, 2)
        y, log_det = transform(x)
        self.assertTrue(torch.allclose(x, y))
        self.assertTrue(torch.allclose(log_det, torch.zeros_like(x)+torch.ones_like(x)))


class TestCouplingTransform(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    def test_AffineCouplingTransform(self):

        def create_resnet(input_dim, output_dim):
            return ResidualNet(
                input_dim,
                output_dim,
                hidden_dim=64,
                num_blocks=2,
                activation=F.relu,
                dropout_probability=0.0,
            )


        transform = AffineCouplingTransform(
            transform_net_create_fn=create_resnet,
            mask=torch.tensor([0, 1, 0, 1], dtype=torch.bool),
        )

        x = torch.randn(32, 4)
        y, log_det = transform(x)
        self.assertTrue(torch.allclose(x[:, 0::2], y[:, 0::2]))
        self.assertFalse(torch.allclose(x[:, 1::2], y[:, 1::2]))

        
        x = torch.randn(32, 4)
        y, log_det = transform.inverse(x)
        self.assertTrue(torch.allclose(x[:, 0::2], y[:, 0::2]))
        self.assertFalse(torch.allclose(x[:, 1::2], y[:, 1::2]))

        x = torch.randn(32, 4)
        y1, log_det1 = transform(x)
        y2, log_det2 = transform.inverse(y1)
        self.assertTrue(torch.allclose(x, y2))
        self.assertTrue(torch.allclose(log_det1, -log_det2))

        x = torch.randn(32, 4)
        y1, log_det1 = transform.inverse(x)
        y2, log_det2 = transform(y1)
        self.assertTrue(torch.allclose(x, y2))
        self.assertTrue(torch.allclose(log_det1, -log_det2))

    def test_AdditiveCouplingTransform(self):

        def create_resnet(input_dim, output_dim):
            return ResidualNet(
                input_dim,
                output_dim,
                hidden_dim=64,
                num_blocks=2,
                activation=F.relu,
                dropout_probability=0.0,
            )

        transform = AdditiveCouplingTransform(
            transform_net_create_fn=create_resnet,
            mask=torch.tensor([0, 1, 0, 1], dtype=torch.bool),
        )

        x = torch.randn(32, 4)
        y, log_det = transform(x)
        self.assertTrue(torch.allclose(x[:, 0::2], y[:, 0::2]))
        self.assertFalse(torch.allclose(x[:, 1::2], y[:, 1::2]))

        
        x = torch.randn(32, 4)
        y, log_det = transform.inverse(x)
        self.assertTrue(torch.allclose(x[:, 0::2], y[:, 0::2]))
        self.assertFalse(torch.allclose(x[:, 1::2], y[:, 1::2]))

        x = torch.randn(32, 4)
        y1, log_det1 = transform(x)
        y2, log_det2 = transform.inverse(y1)
        self.assertTrue(torch.allclose(x, y2))
        self.assertTrue(torch.allclose(log_det1, -log_det2))

        x = torch.randn(32, 4)
        y1, log_det1 = transform.inverse(x)
        y2, log_det2 = transform(y1)
        self.assertTrue(torch.allclose(x, y2))
        self.assertTrue(torch.allclose(log_det1, -log_det2))



if __name__ == '__main__':

    def create_resnet(input_dim, output_dim):
        return ResidualNet(
            input_dim,
            output_dim,
            hidden_dim=64,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0.0,
        )

    transform = AdditiveCouplingTransform(
        transform_net_create_fn=create_resnet,
        mask=torch.tensor([0, 1, 0, 1], dtype=torch.bool),
    )
    x = torch.randn(32, 4)
    y, log_det = transform(x)
    b=1
