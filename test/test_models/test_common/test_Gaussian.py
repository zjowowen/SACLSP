# make a unittest for Gaussian and GaussianTanh

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from SACLSP.models.common import Gaussian, GaussianTanh


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.dim = 2
        self.model = Gaussian(nn.Linear(self.dim, self.dim * 2))

    def test_forward(self):
        obs = torch.randn(self.batch_size, self.dim)
        action, logp = self.model(obs)
        self.assertEqual(action.shape, (self.batch_size, self.dim))
        self.assertEqual(logp.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(action, self.model.mean(obs)))
        self.assertTrue(torch.allclose(logp, self.model.logstd(obs)))

    def test_entropy(self):
        obs = torch.randn(self.batch_size, self.dim)
        entropy = self.model.entropy(obs)
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(entropy, self.model.entropy(obs)))

    def test_sample(self):
        obs = torch.randn(self.batch_size, self.dim)
        action = self.model.sample(obs)
        self.assertEqual(action.shape, (self.batch_size, self.dim))
        self.assertTrue(torch.allclose(action, self.model.sample(obs)))

    def test_logprob(self):
        obs = torch.randn(self.batch_size, self.dim)
        action = torch.randn(self.batch_size, self.dim)
        logprob = self.model.logprob(obs, action)
        self.assertEqual(logprob.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(logprob, self.model.logprob(obs, action)))


class TestGaussianTanh(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.dim = 2
        self.model = GaussianTanh(nn.Linear(self.dim, self.dim * 2))

    def test_forward(self):
        obs = torch.randn(self.batch_size, self.dim)
        action, logp = self.model(obs)
        self.assertEqual(action.shape, (self.batch_size, self.dim))
        self.assertEqual(logp.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(action, self.model.mean(obs)))
        self.assertTrue(torch.allclose(logp, self.model.logstd(obs)))

    def test_entropy(self):
        obs = torch.randn(self.batch_size, self.dim)
        entropy = self.model.entropy(obs)
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(entropy, self.model.entropy(obs)))

    def test_sample(self):
        obs = torch.randn(self.batch_size, self.dim)
        action = self.model.sample(obs)
        self.assertEqual(action.shape, (self.batch_size, self.dim))
        self.assertTrue(torch.allclose(action, self.model.sample(obs)))

    def test_logprob(self):
        obs = torch.randn(self.batch_size, self.dim)
        action = torch.randn(self.batch_size, self.dim)
        logprob = self.model.logprob(obs, action)
        self.assertEqual(logprob.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(logprob, self.model.logprob(obs, action)))

    
