import torch
import torch.nn as nn
import torch.nn.functional as F


def unified_loss(model, x_0, ts, noise_scheduler, samples_cond=None):
  '''
  a unified loss function for score-based generative models
  model: the score model to be trained
  '''
  marg_mean,marg_std = noise_scheduler.marginal_prob(x_0, ts)
  z = torch.randn_like(x_0)
  perturbed_samples = marg_mean + z * marg_std

  if samples_cond is None:
      scores = model(perturbed_samples, ts)
  else:
      scores = model(perturbed_samples, samples_cond, ts)

  loss = torch.square(scores * marg_std + z).mean()

  return loss


#lose for ncsn
def anneal_dsm_score_estimation(model, x_0, ts, noise_scheduler, anneal_power=2., samples_cond=None):

    perturbed_samples,used_sigmas = noise_scheduler.forward_sample(x_0, ts)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - x_0)

    if samples_cond is None:
        scores = model(perturbed_samples, ts)
    else:
        scores = model(perturbed_samples, samples_cond, ts)

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * \
        used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)

#loss for ddpm
def noise_estimation_loss(model, x_0, ts, noise_scheduler,x_cond=None):

    x,e = noise_scheduler.forward_sample(x_0, ts)

    if x_cond is None:
        output = model(x, ts)
    else:
        output = model(x,x_cond, ts)

    return (e - output).square().mean()

# loss for sde
def sde_score_matching_loss(model, x, sde, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  _,std = sde.marginal_prob(x,random_t)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score* std[:, None] + z)**2, dim=(1)))
  return loss