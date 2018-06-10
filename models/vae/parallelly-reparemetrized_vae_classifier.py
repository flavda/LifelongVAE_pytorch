from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

from models.reparameterizers.gumbel import GumbelSoftmax
from models.reparameterizers.mixture import Mixture
from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from models.vae.abstract_vae import AbstractVAE


class ParallellyReparameterizedVAEClassifier(ParallellyReparameterizedVAE):
    ''' This implementation uses a parallel application of
        the reparameterizer via the mixture type. '''
    def __init__(self, input_shape, activation_fn=nn.ELU, num_current_model=0, **kwargs):
        super(ParallellyReparameterizedVAEClassifier, self).__init__(input_shape,
                                                           activation_fn=activation_fn,
                                                           num_current_model=num_current_model,
                                                           **kwargs)
        self.output_shape =  output_shape



        # build classifier
      #  self.classifier = self.build_classifier()

    def build_dense_classifier(self, input_shape, output_size, latent_size=512, activation_fn=nn.ELU):
        input_flat = int(np.prod(input_shape))
        output_flat = int(np.prod(output_size))
        output_size = [output_size] if not isinstance(output_size, list) else output_size
        return nn.Sequential(
            View([-1, input_flat]),
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            activation_fn(),
            nn.Linear(latent_size, output_flat),
            View([-1, *output_size])
        )

    def build_classifier(self):
        ''' helper function to build dense classifier'''

        classifier = self.build_dense_classifier(input_size=self.reparameterizer.output_size,
                                            output_shape=10,
                                            activation_fn=self.activation_fn,
                                            normalization_str=self.config['normalization'])
        if self.config['ngpu'] > 1:
            classifier = nn.DataParallel(classifier)

        if self.config['cuda']:
            classifier = classifier.cuda()

        return classifier

    def classify(self, x):
   # ''' classify latent variables z '''
        z_logits = self.encode(x)
        return self.build_classifier(z_logits)

    def forward(self, x, y):
        ''' params is a map of the latent variable's parameters'''
        z, params = self.posterior(x)

        return self.decode(z), params, self.classifier(x)

    def loss_function(self, recon_x, x, params, y_hat, y, mut_info = None):
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        kld = self.config['kl_reg'] * self.kld(params)
        elbo = nll + kld
        classification_loss = F.cross_entropy(y_hat, y)

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
                                                                                           mut_info=mut_info)
