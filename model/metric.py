import torch
import torch.nn.functional as F
import probtorch.objectives.montecarlo as objectives

import base.base_model as base

def elbo(output, target):
    p, q, _ = output

    return -objectives.elbo(q, p, sample_dim=None, batch_dim=0)

def log_likelihood(output, target, metadata=None):
    p, q, _ = output

    return -objectives.log_like(q, p, sample_dim=None, batch_dim=0)

def reconstruction_error(output, target, metadata=None):
    p, q, _ = output

    reconstruction = p['reconstruction'].dist.probs
    observation = p['reconstruction'].value
    cross_entropy = base.probtorch_cross_entropy(reconstruction, observation)
    return cross_entropy.mean(dim=0)

def _sample_var_batch(metadata, scale, B=20):
    bX, by = [], []
    for _ in range(B):
        xi, factor = metadata['dataset'].sample_fixed_factor(size=200)
        xi = torch.Tensor(xi).to(metadata['device']).transpose(-3, -1)
        p, _, _ = metadata['model'].forward(xi)
        zi = p['z'].dist.loc / scale
        D = torch.var(zi, dim=0).argmin().item()
        f = factor - metadata['dataset'].diff
        bX.append(D)
        by.append(f)
    return torch.tensor(bX), torch.tensor(by)

def disentanglement_metric(output, target, metadata):
    p, q, _ = output

    scale = p['z'].value.std(dim=0)
    batch_imgs, batch_classes = _sample_var_batch(metadata, scale, B=800)
    V = torch.zeros(metadata['model'].z_dim, metadata['dataset'].n_factors)
    for imgs, targets in zip(batch_imgs, batch_classes):
        V[imgs, targets] += 1

    target_imgs, target_classes = _sample_var_batch(metadata, scale, B=800)
    V_targets = torch.stack([V[target] for target in target_imgs], dim=0)
    preds = torch.argmax(V_targets, dim=-1)

    return torch.mean((preds == target_classes).to(dtype=torch.float), dim=0)
