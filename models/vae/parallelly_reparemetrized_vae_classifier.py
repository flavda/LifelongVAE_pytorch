from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from models.reparameterizers.gumbel import GumbelSoftmax
from models.reparameterizers.mixture import Mixture
from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from models.vae.abstract_vae import AbstractVAE
from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from helpers.layers import View, build_dense_encoder, build_dense_decoder, flatten_layers, Identity
#from helpers.layers_classify import build_dense_classifier
from helpers.utils import float_type, long_type
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn


class ParallellyReparameterizedVAEClassifier(ParallellyReparameterizedVAE):
    ''' This implementation uses a parallel application of
        the reparameterizer via the mixture type. '''
    def __init__(self, input_shape, output_size=10, activation_fn=nn.ELU, num_current_model=0, **kwargs):
        super(ParallellyReparameterizedVAEClassifier, self).__init__(input_shape,
                                                                     activation_fn=activation_fn,
                                                                     num_current_model=num_current_model,
                                                           **kwargs)
        self.output_size = output_size
        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            print("using gumbel softmax reparameterizer")
            self.reparameterizer = GumbelSoftmax(self.config)
        elif self.config['reparam_type'] == "mixture":
            print("using mixture reparameterizer")
            self.reparameterizer = Mixture(num_discrete=self.config['discrete_size'],
                                           num_continuous=self.config['continuous_size'],
                                           config=self.config)
        else:
            raise Exception("unknown reparameterization type")


     # build classifier which will classify latent variables z
        self.classifier = self.build_classifier()

        # build a classifier which will classify x
        self.simple_classifier = self.build_simlpe_classifier()


    def build_classifier(self):

        classifier = build_dense_encoder(input_shape=self.reparameterizer.output_size,
                                            output_size=10, latent_size=512,
                         activation_fn=nn.ELU, normalization_str=self.config['normalization_classifier'])


        if self.config['ngpu'] > 1:
            classifier = nn.DataParallel(classifier)

        if self.config['cuda']:
            classifier = classifier.cuda()

        return classifier


    def classify(self, z):
   # ''' classify latent variables z '''
    #    return self.build_classifier(z_logits)
        return self.classifier(z)

    def build_simlpe_classifier(self):

        simple_classifier = build_dense_encoder(input_shape = self.input_shape,
                                                output_size=10, latent_size=512,
                         activation_fn=nn.ELU, normalization_str=self.config['normalization_classifier'])


        if self.config['ngpu'] > 1:
            simple_classifier = nn.DataParallel(simple_classifier)

        if self.config['cuda']:
            simple_classifier = simple_classifier.cuda()

        return simple_classifier

    def simple_classify(self, x):
        # ''' classify input variables x '''
        return self.simple_classifier(x)

    def forward(self, x, y):
        ''' params is a map of the latent variable's parameters'''
        z, params = self.posterior(x)
        #return self.decode(z), params, self.classify(z)

        if self.config['disable_VAEclassifier'] is True:
            # simple/traditional classification based on input x
            return self.decode(z), params, self.simple_classify(x)
        # cassification based on lattent variables z
        return self.decode(z), params, self.classify(z)



    def loss_function(self, recon_x, x, params, y_hat_logits, y, mut_info = None):
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        kld = self.config['kl_reg'] * self.kld(params)
        elbo = nll + kld
        classification_loss = F.cross_entropy(y_hat_logits, y, reduce=False)

        mut_info = self.mut_info(params)
        # handle the mutual information term
        if mut_info is None:
            mut_info = Variable(
                float_type(self.config['cuda'])(x.size(0)).zero_()
            )
        else:
            # Clamping strategies
            mut_clamp_strategy_map = {
                'none': lambda mut_info: mut_info,
                'norm': lambda mut_info: mut_info / torch.norm(mut_info, p=2),
                'clamp': lambda mut_info: torch.clamp(mut_info,
                                                      min=-self.config['mut_clamp_value'],
                                                      max=self.config['mut_clamp_value'])
            }
            mut_info = mut_clamp_strategy_map[self.config['mut_clamp_strategy'].strip().lower()](mut_info)


        loss = elbo + classification_loss - mut_info

        return {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'mut_info_mean': torch.mean(mut_info),
            'classification_loss_mean': torch.mean(classification_loss)
        }



 #   def loss_function(self, recon_x, x, params, y_hat, y):
  #      ''' evaluates the loss of the model '''
   #     mut_info = self.mut_info(params)
#
 #       return super(ParallellyReparameterizedVAEClassifier,self).loss_function(recon_x, x, params, y_hat, y,
 #                                                                                          mut_info=mut_info)
