import torch
import torch.nn as nn
import torch.nn.functional as F


def anneal_dsm_score_estimation(model, samples, labels, sigmas, anneal_power=2., samples_cond=None):
    used_sigmas = sigmas[labels].view(
        samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)

    if samples_cond is None:
        scores = model(perturbed_samples, labels)
    else:
        scores = model(perturbed_samples, samples_cond, labels)

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * \
        used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)
