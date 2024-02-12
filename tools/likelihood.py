import torch
import numpy as np
from scipy import integrate
# from models import utils as mutils

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn

class ode_likelihood():

    def __init__(self,sde, score_fn,  device='cuda'):
        self.sde = sde
        self.score_fn = score_fn
        self.device = device

    def drift_fn(self, x, t):
        """Get the drift function of the reverse-time SDE."""

        rsde = self.sde.reverse(self.score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(self, x, t, noise):
        return get_div_fn(lambda xx, tt: self.drift_fn(xx, tt))(x, t, noise)

    def likelihood_eval(self, data,eps=1e-5,rtol=1e-5, atol=1e-5,
                    method='RK45',hutchinson_type='Rademacher'):

        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(self.drift_fn(sample, vec_t))
                logp_grad = to_flattened_numpy(self.div_fn(sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, self.sde.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            prior_logp = self.sde.prior_logp(z)
            totoal_logp = prior_logp + delta_logp
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N + 8.

        return bpd,totoal_logp