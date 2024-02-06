
import torch
import numpy as np
# import sde_lib
import abc
from scipy import integrate

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None] * np.sqrt(-dt) * z
    return x, x_mean

class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None] * z
    return x, x_mean



class LangevinCorrector():
  def __init__(self, sde, score_fn, snr, n_steps):
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps
    # super().__init__(sde, score_fn, snr, n_steps)
    # if not isinstance(sde, sde_lib.VPSDE) \
    #     and not isinstance(sde, sde_lib.VESDE) \
    #     and not isinstance(sde, sde_lib.subVPSDE):
    #   raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    # if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    #   timestep = (t * (sde.N - 1) / sde.T).long()
    #   alpha = sde.alphas.to(t.device)[timestep]
    # else:
    #   alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2*torch.ones_like(t)
      x_mean = x + step_size[:, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise

    return x, x_mean


class PC_sampler():
  def __init__(self, sde, score_fn, snr, n_correct_steps,predictor='EulerMaruyama',corrector='Langevin',device='cuda'):
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_correct_steps
    self.device=device

    if predictor == 'EulerMaruyama':
      self.predictor = EulerMaruyamaPredictor(sde, score_fn)
    elif predictor == 'ReverseDiffusion':
      self.predictor = ReverseDiffusionPredictor(sde, score_fn)
    else:
        raise NotImplementedError(f"Predictor {predictor} not yet supported.")
    
    if corrector == 'Langevin':
      self.corrector = LangevinCorrector(sde, score_fn, snr, n_correct_steps)
    else:
        raise NotImplementedError(f"Corrector {corrector} not yet supported.")

  def sample(self, batch_size, eps=1e-5):
    with torch.no_grad():
      # Initial sample
      x = self.sde.prior_sampling((batch_size,2)).to(self.device)
      timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=self.device)

      for i in range(self.sde.N):
        t = timesteps[i]
        vec_t = torch.ones(batch_size, device=t.device) * t
        x, x_mean = self.corrector.update_fn(x, vec_t)
        x, x_mean = self.predictor.update_fn(x, vec_t)

    return x_mean


class ode_sampler():
    def __init__(self,sde, score_fn,  device='cuda'):
        self.sde = sde
        self.score_fn = score_fn
        self.device = device

    # def denoise_update_fn(self, x):
    #     # Reverse diffusion predictor for denoising
    #     predictor_obj = ReverseDiffusionPredictor(self.sde, self.score_fn, probability_flow=False)
    #     vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    #     _, x = predictor_obj.update_fn(x, vec_eps)
    #     return x

    def drift_fn(self, x, t):
        """Get the drift function of the reverse-time SDE."""
 
        rsde = self.sde.reverse(self.score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def sample(self, batch_size,z=None,eps=1e-5,rtol=1e-5, atol=1e-5,
                    method='RK45'):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
        model: A score model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = self.sde.prior_sampling((batch_size,2)).to(self.device)
            else:
                x = z

            def ode_func(t, x):
                x = torch.tensor(x.reshape(batch_size,2)).to(self.device).type(torch.float32)
                vec_t = torch.ones(batch_size, device=x.device) * t
                drift = self.drift_fn(x, vec_t)
                return drift.cpu().numpy().reshape((-1,)).astype(np.float64)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (self.sde.T, eps), x.cpu().numpy().reshape((-1,)),
                                            rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape((batch_size,2)).to(self.device).type(torch.float32)

        return x, nfe
    
