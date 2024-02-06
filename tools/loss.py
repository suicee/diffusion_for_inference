import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # Select a random step for each example
    # t = torch.randint(0, noise_scheduler.n_steps, size=(batch_size // 2 + 1,))
    # t = torch.cat([t, noise_scheduler.n_steps - t - 1], dim=0)[:batch_size].long().to(x_0.device)
    # # x0 multiplier
    # a = extract(alphas_bar_sqrt, t, x_0).to(x_0.device)
    # # eps multiplier
    # am1 = extract(one_minus_alphas_bar_sqrt, t, x_0).to(x_0.device)
    # e = torch.randn_like(x_0).to(x_0.device)
    # # model input
    # x = x_0 * a + e * am1
    # print(x_0.shape,ts.shape)
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