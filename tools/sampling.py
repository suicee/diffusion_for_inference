import torch
import math


def annealed_langevin_dynamic(sigma_min, sigma_max, n_steps, annealed_step, score_fn,init_point, device,sample_cond=None, eps=1e-1, only_final=False):
    process = torch.exp(torch.linspace(start=math.log(
        sigma_max), end=math.log(sigma_min), steps=n_steps)).to(device=device)
    step_size = eps * (process / process[-1]) ** 2
    # step_size = torch.ones_like(process) * eps
    print(step_size)
    sample = init_point
    sampling_list = []

    final = None
    score_fn.eval()
    for idx in range(len(process)):
        labels = torch.ones(init_point.shape[0], dtype=torch.long).to(device=device) * idx
        for _ in range(annealed_step):
            z, step = torch.randn_like(sample).to(
                device=device), step_size[idx]
            with torch.no_grad():
                if sample_cond is None:
                    sample = sample + 0.5 * step * \
                        score_fn(sample, labels) + torch.sqrt(step) * z
                else: 
                    sample = sample + 0.5 * step * \
                        score_fn(sample, sample_cond, labels) + torch.sqrt(step) * z

        final = sample
        if not only_final:
            sampling_list.append(final)

    return final if only_final else torch.stack(sampling_list)