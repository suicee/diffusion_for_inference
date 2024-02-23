import torch
import math
import numpy as np

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class annealed_langevin_dynamic_sampler():
    def __init__(self, score_fn, noise_scheduler , n_iter_each_T , device, eps = 1e-1):
        '''
        sigma_min : minimum sigmas of perturbation schedule 
        sigma_max : maximum sigmas of perturbation schedule 
        n_iter_each_T         : iteration step of Langevin dynamic at each temperature level
        score_fn  : trained score network
        eps       : coefficient of step size
        '''
        self.sigmas = noise_scheduler.marg_std
        self.n_steps = len(self.sigmas)
        
        self.step_size = eps * (self.sigmas / self.sigmas[0] ) ** 2

        self.score_fn = score_fn
        self.score_fn.eval()

        self.annealed_step = n_iter_each_T
        self.device = device

    def _one_annealed_step_iteration(self, x, idx, sample_cond=None):
        '''
        x   : perturbated data
        idx : step of perturbation schedule
        '''
        z, step_size = torch.randn_like(x).to(device = self.device), self.step_size[idx]
        labels = torch.ones(x.shape[0], dtype=torch.long).to(device=self.device) * idx

        if sample_cond is None:
            x = x + 0.5 * step_size * self.score_fn(x, labels) + torch.sqrt(step_size) * z
        else:
            x = x + 0.5 * step_size * self.score_fn(x,sample_cond, labels) + torch.sqrt(step_size) * z

        return x

    @torch.no_grad()
    def one_temperature_sampling(self,init_sample, idx, sample_cond=None):
        '''
        x   : perturbated data
        idx : step of perturbation schedule
        '''
        x = init_sample
        for _ in range(self.annealed_step):
            x = self._one_annealed_step_iteration(x, idx, sample_cond)
        return x
    
    @torch.no_grad()
    def loop_sample(self, shape,sample_cond=None):
        '''
        only_final : If True, return is an only output of final schedule step 
        '''
        cur_x = torch.randn(shape).to(device=self.device)
        x_seq = [cur_x]
        
        for idx in reversed(range(len(self.sigmas))):
            cur_x = self.one_temperature_sampling(cur_x, idx, sample_cond)
            x_seq.append(cur_x)
                

        return x_seq


class ddpm_sampler():

    def __init__(self, score_fn, noise_scheduler, device):
        self.score_fn = score_fn
        self.score_fn.eval()
        self.score_fn.to(device)
        # self.sigmas=sigmas

        self.betas = noise_scheduler.betas
        self.alphas = noise_scheduler.alphas
        # alphas = 1 - self.betas
        # alphas_prod = torch.cumprod(alphas, 0)
        # one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        # self.alphas=alphas
        # self.one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt       
        self.n_steps = len(self.betas)
        self.device=device

    # def one_level_sample(self, x, t,sample_cond=None):
    #     t = torch.tensor([t]).to(self.device)
    #     # Factor to the model output
    #     eps_factor = ((1 - extract(self.alphas, t, x)) / extract(self.one_minus_alphas_bar_sqrt, t, x)).to(self.device)
    #     # Model output
    #     if sample_cond is None:
    #         eps_theta = self.model(x, t)
    #     else:
    #         eps_theta = self.model(x, sample_cond, t)
    #     # Final values
    #     mean = ((1 / extract(self.alphas, t, x).sqrt()).to(self.device) * (x - (eps_factor * eps_theta)))
    #     # Generate z
    #     z = torch.randn_like(x).to(self.device)
    #     # Fixed sigma
    #     # sigma_t = extract(self.betas, t, x).sqrt().to(self.device)
    #     sigma_t = extract(self.sigmas, t, x).to(self.device)
    #     sample = mean + sigma_t * z
    #     return sample
        
    @torch.no_grad()
    def one_denoise_step(self, x, t,sample_cond=None):

        t = torch.tensor([t]).to(self.device)

        if sample_cond is None:
            score = self.score_fn(x, t)
        else:
            score = self.score_fn(x, sample_cond, t)

        mean = ((1 / extract(self.alphas, t, x).sqrt()).to(self.device) * (x + extract(self.betas, t, x).to(self.device) * score))

        z = torch.randn_like(x).to(self.device)
        sigma_t = extract(self.betas, t, x).sqrt().to(self.device)
        sample = mean + sigma_t * z
        return sample,mean
    @torch.no_grad()
    def loop_sample(self, shape,sample_cond=None):
        cur_x = torch.randn(shape).to(device=self.device)
        x_seq = [cur_x]
        for i in reversed(range(self.n_steps)):
            cur_x,cur_x_mean = self.one_denoise_step(cur_x, i,sample_cond)
            x_seq.append(cur_x)
        x_seq.append(cur_x_mean)
        return x_seq

class ddim_sampler():
    def __init__(self, score_fn, noise_scheduler, eta,tau=1, device='cuda', scheduling = 'uniform'):
        self.score_fn = score_fn
        self.score_fn.eval()
        self.score_fn.to(device)

        self.betas = noise_scheduler.betas
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])

        self.sigmas = torch.sqrt((1 - self.alpha_prev_bars) / (1 - self.alpha_bars)) * torch.sqrt(1 - (self.alpha_bars / self.alpha_prev_bars))

        self.device=device
        self.eta = eta
        self.tau = tau
        self.scheduling = scheduling
    
    def _get_process_scheduling(self, reverse = True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alpha_bars), self.tau)) + [len(self.alpha_bars)-1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alpha_bars)* 0.8), self.tau)** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alpha_bars)-1]
        else:
            assert 'Not Implementation'
            
        
        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process
    @torch.no_grad()
    def one_denoise_step(self, x,prev_idx, idx,sample_cond=None):

        noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
        labels = torch.ones(x.shape[0]).long().to(self.device) * idx
        if sample_cond is not None:
            score = self.score_fn(x, labels, sample_cond)
        else:
            score = self.score_fn(x, labels)

        sigma = self.sigmas[idx] * self.eta
        #transform score to predict epsilon
        predict_epsilon = -score*torch.sqrt(1 - self.alpha_bars[idx])
        
        predicted_x0 = torch.sqrt(self.alpha_bars[prev_idx]) * (x - torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_bars[idx])
        direction_pointing_to_xt = torch.sqrt(1 - self.alpha_bars[prev_idx] - sigma**2 ) * predict_epsilon
        x = predicted_x0 + direction_pointing_to_xt + sigma * noise

        return x
    @torch.no_grad()
    def loop_sample(self, shape,sample_cond=None):
        cur_x = torch.randn(shape).to(device=self.device)
        diffusion_process = self._get_process_scheduling(reverse = True)
        x_seq = [cur_x]
        for prev_idx, idx in diffusion_process:
            cur_x = self.one_denoise_step(cur_x,prev_idx, idx,sample_cond=sample_cond)
            x_seq.append(cur_x)
        
        return x_seq


        


# def annealed_langevin_dynamic(sigma_min, sigma_max, n_steps, annealed_step, score_fn,init_point, device,sample_cond=None, eps=1e-1, only_final=False):
#     process = torch.exp(torch.linspace(start=math.log(
#         sigma_max), end=math.log(sigma_min), steps=n_steps)).to(device=device)
#     step_size = eps * (process / process[-1]) ** 2
#     # step_size = torch.ones_like(process) * eps
#     print(step_size)
#     sample = init_point
#     sampling_list = []

#     final = None
#     score_fn.eval()
#     for idx in range(len(process)):
#         labels = torch.ones(init_point.shape[0], dtype=torch.long).to(device=device) * idx
#         for _ in range(annealed_step):
#             z, step = torch.randn_like(sample).to(
#                 device=device), step_size[idx]
#             with torch.no_grad():
#                 if sample_cond is None:
#                     sample = sample + 0.5 * step * \
#                         score_fn(sample, labels) + torch.sqrt(step) * z
#                 else: 
#                     sample = sample + 0.5 * step * \
#                         score_fn(sample, sample_cond, labels) + torch.sqrt(step) * z

#         final = sample
#         if not only_final:
#             sampling_list.append(final)

#     return final if only_final else torch.stack(sampling_list)