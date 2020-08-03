import torch.nn.functional as F
import probtorch.objectives.montecarlo as objectives

import base.base_model as base

def elbo(output, target):
    p, q, _ = output

    return -objectives.elbo(q, p, sample_dim=None, batch_dim=0)

def log_likelihood(output, target):
    p, q, _ = output

    return -objectives.log_like(q, p, sample_dim=None, batch_dim=0)

def reconstruction_error(output, target):
    p, q, _ = output

    reconstruction = p['reconstruction'].dist.probs
    observation = p['reconstruction'].value
    cross_entropy = base.probtorch_cross_entropy(reconstruction, observation)
    return cross_entropy.mean(dim=0)
