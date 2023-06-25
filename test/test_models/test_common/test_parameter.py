# make a unittest for NonegativeParameter, TanhParameter  and CovarianceMatrix


import unittest

import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict

from SACLSP.models.common import NonegativeParameter, TanhParameter, CovarianceMatrix


class TestNonegativeParameter(unittest.TestCase):

    def setUp(self):
        self.dim = [10, 2]
        self.data = torch.exp(nn.init.normal_(torch.Tensor(self.dim)))
        self.param = NonegativeParameter(data=self.data)

    def test_forward(self):
        self.assertEqual(self.param.forward().shape, self.data.shape)
        self.assertTrue(torch.allclose(self.param.forward(), self.data))

    def test_data(self):
        self.assertEqual(self.param.data.shape, self.data.shape)
        self.assertTrue(torch.allclose(self.param.data, self.data))


class TestTanhParameter(unittest.TestCase):

    def setUp(self):
        self.dim = [10, 2]
        self.data=torch.tanh(nn.init.normal_(torch.Tensor(self.dim)))
        self.param = TanhParameter(data=self.data)

    def test_forward(self):
        self.assertEqual(self.param.forward().shape, self.data.shape)
        self.assertTrue(torch.allclose(self.param.forward(), self.data))

    def test_data(self):
        self.assertEqual(self.param.data.shape, self.data.shape)
        self.assertTrue(torch.allclose(self.param.data, self.data))    


class TestCovarianceMatrix(unittest.TestCase):

    def setUp(self):
        self.dim = 3
        cfg=dict(
                dim=self.dim,
                functional=False,
                random_init=False,
                sigma_lambda=dict(
                    hidden_sizes=[11, 128, 128],
                    activation='relu',
                    output_size=self.dim,
                    dropout=0,
                    layernorm=False,
                ),
                sigma_offdiag=dict(
                    hidden_sizes=[11, 128, 128],
                    activation='relu',
                    output_size=self.dim*(self.dim-1)//2,
                    dropout=0,
                    layernorm=False,
                ),
            )
        cfg=EasyDict(cfg)
        self.model = CovarianceMatrix(cfg)

    def test_low_triangle_matrix(self):
        self.assertEqual(self.model.low_triangle_matrix().shape, torch.Tensor(self.dim, self.dim).shape)
        self.assertEqual(self.model.forward().shape, torch.Tensor(self.dim, self.dim).shape)
        self.assertTrue(torch.linalg.det(self.model.forward())>0.0)

