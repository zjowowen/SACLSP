# make a unittest for Gaussian and GaussianTanh

import unittest

import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict

from SACLSP.models.common import Gaussian, GaussianTanh, StandardGaussian


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.input_dim = 2
        self.output_dim = 3
        cfg=dict(
            mu_model=dict(
                hidden_sizes=[self.input_dim, 64],
                activation='softplus',
                layernorm=False,
                output_size=self.output_dim,
            ),
            cov=dict(
                dim=self.output_dim,
                functional=True,
                random_init=False,
                sigma_lambda=dict(
                    hidden_sizes=[self.input_dim, 128, 128],
                    activation='relu',
                    output_size=self.output_dim,
                ),
                sigma_offdiag=dict(
                    hidden_sizes=[self.input_dim, 128, 128],
                    activation='relu',
                    output_size=self.output_dim,
                ),
            ),
        )
        cfg=EasyDict(cfg)
        self.model = Gaussian(cfg)

    def test_forward(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action, logp = self.model(obs)
        self.assertEqual(action.shape, (self.batch_size, self.output_dim))
        self.assertEqual(logp.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(logp, self.model.log_prob(action,obs)))

    def test_entropy(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        entropy = self.model.entropy(obs)
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(entropy, self.model.entropy(obs)))

    def test_sample(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action = self.model.sample(obs)
        self.assertEqual(action.shape, (self.batch_size, self.output_dim))

    def test_logprob(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action = torch.randn(self.batch_size, self.output_dim)
        logprob = self.model.log_prob(action, obs)
        self.assertEqual(logprob.shape, (self.batch_size,))


class TestGaussianTanh(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.input_dim = 2
        self.output_dim = 3
        cfg=dict(
            mu_model=dict(
                hidden_sizes=[self.input_dim, 64],
                activation='softplus',
                layernorm=False,
                output_size=self.output_dim,
            ),
            cov=dict(
                dim=self.output_dim,
                functional=True,
                random_init=False,
                sigma_lambda=dict(
                    hidden_sizes=[self.input_dim, 128, 128],
                    activation='relu',
                    output_size=self.output_dim,
                ),
                sigma_offdiag=dict(
                    hidden_sizes=[self.input_dim, 128, 128],
                    activation='relu',
                    output_size=self.output_dim,
                ),
            ),
        )
        cfg=EasyDict(cfg)
        self.model = GaussianTanh(cfg)

    def test_forward(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action, logp = self.model(obs)
        self.assertEqual(action.shape, (self.batch_size, self.output_dim))
        self.assertEqual(logp.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(logp, self.model.log_prob(action, obs),rtol=1e-04, atol=1e-05))

    def test_sample(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action = self.model.sample(obs)
        self.assertEqual(action.shape, (self.batch_size, self.output_dim))

    def test_logprob(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action = torch.tanh(torch.randn(self.batch_size, self.output_dim))
        logprob = self.model.log_prob(action, obs)
        self.assertEqual(logprob.shape, (self.batch_size,))


class TestStandardGaussian(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.input_dim = 2
        self.output_dim = 3
        self.model = StandardGaussian(self.output_dim)

    def test_entropy(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        entropy = self.model.entropy()
        self.assertTrue(torch.allclose(entropy, self.model.entropy()))
    
    def test_sample(self):
        action = self.model.sample()
        self.assertEqual(action.shape, (self.output_dim, ))

    def test_logprob(self):
        obs = torch.randn(self.batch_size, self.input_dim)
        action = torch.randn(self.batch_size, self.output_dim)
        logprob = self.model.log_prob(action, obs)
        self.assertEqual(logprob.shape, (self.batch_size,))
