from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, zeros_like, ones_like
from models.layers import View, Identity


class IsotropicGaussian(nn.Module):
    ''' isotropic gaussian reparameterization '''
    def __init__(self, config):
        super(IsotropicGaussian, self).__init__()
        self.config = config
        self.input_size = self.config['continuous_size']
        assert self.config['continuous_size'] % 2 == 0
        self.output_size = self.config['continuous_size'] // 2

    def prior(self, shape):
        return Variable(
            float_type(self.config['cuda'])(*shape).normal_()
        )

    def _reparametrize_gaussian(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = float_type(self.config['cuda'])(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu), {'mu': mu, 'logvar': logvar}

    def reparmeterize(self, logits):
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0
        mu = logits[:, 0:int(feature_size/2)]
        sigma = F.softplus(logits[:, int(feature_size/2):])
        #sigma = logits[:, int(feature_size/2):]
        return self._reparametrize_gaussian(mu, sigma)

    @staticmethod
    def _kld_gaussian_N_0_1(mu, logvar, cuda=False):
        standard_normal = D.Normal(zeros_like(mu, cuda),
                                   ones_like(logvar, cuda))
        normal = D.Normal(mu, logvar)
        return torch.sum(D.kl_divergence(normal, standard_normal), dim=-1)

    def kl(self, dist_a):
        return IsotropicGaussian._kld_gaussian_N_0_1(
            dist_a['gaussian']['mu'], dist_a['gaussian']['logvar'],
            cuda=self.config['cuda']
        )

    def forward(self, logits):
        z, gauss_params = self.reparmeterize(logits)
        return z, { 'z': z, 'gaussian':  gauss_params }
