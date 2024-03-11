import torch
from matplotlib import pyplot as plt
import logging
import math
logging.basicConfig(level=logging.INFO)



def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    codes from diffusers package that implements the beta schedule in http://arxiv.org/abs/2102.09672

    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

class noise_scheduler():
    def __init__(self, beta_start, beta_end, n_steps,var_norm=True,schedule='linear'):
        '''
        beta_start : beta value from t=0 to t=1
        beta_end   : beta value from t=T-1 to t=T
        n_steps    : number of T
        var_norm   : If True, use variance normalization(DDPM), else do not use variance normalization (NCSN)
        schedule   : how to interpolate beta from beta_start to beta_end
        '''

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_steps = n_steps
        self.schedule = schedule
        self.var_norm = var_norm

        betas=self.make_beta_schedule(schedule=self.schedule,
                                             n_timesteps=self.n_steps, 
                                             start=self.beta_start, 
                                             end=self.beta_end)

        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, 0)
        alphas_bar_sqrt = torch.sqrt(alphas_bar)
        # one_minus_alphas_bar_log = torch.log(1 - alphas_bar)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)

        self.sigmas=torch.sqrt(betas)
        self.alphas=alphas
        self.betas=betas
        self.alphas_bar=alphas_bar
        self.alphas_bar_sqrt=alphas_bar_sqrt
        # self.one_minus_alphas_bar_log=one_minus_alphas_bar_log
        self.one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt

        #std of the conditional distribution p(x_t|x_0)
        if self.var_norm:
            #for ddpm x_t|x_0 ~ N(alphas_bar_sqrt*x_0,1-alphas_bar)
            self.marg_std = self.one_minus_alphas_bar_sqrt
        else:
            #for ncsn x_t|x_0 ~ N(0,(beta1+...+betat))
            self.marg_std = torch.sqrt(torch.cumsum(self.betas, 0))

    @staticmethod
    def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == "cosine":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(n_timesteps)
        return betas

    
    def forward_sample(self,x_0, t):
        '''
        diffusion process
        x_0 : initial data
        t   : time step

        return : x_t
        '''
        noise = torch.randn_like(x_0).to(x_0.device)
        with torch.no_grad():
            if self.var_norm:
                #for ddpm x_t|x_0 ~ N(alphas_bar_sqrt*x_0,1-alphas_bar), x_t = alphas_bar_sqrt*x_0 + sqrt(1-alphas_bar)*noise
                alphas_t = extract(self.alphas_bar_sqrt, t, x_0).to(x_0.device)
                alphas_1_m_t = extract(self.one_minus_alphas_bar_sqrt, t, x_0).to(x_0.device)
                return alphas_t * x_0 + alphas_1_m_t * noise#,noise
            else:
                #for ncsn x_t|x_0 ~ N(0,(beta1+...+betat)), x_t = x_0 + sqrt(betasum)*noise
                used_sigmas=extract(self.marg_std,t,x_0).to(x_0.device)
                return x_0+used_sigmas*noise#,used_sigmas
            
    def marginal_prob(self, x_0, t):
        '''
        x_0 : data
        t : time step

        return : mean, std of p(x_t|x_0)
        '''
        if self.var_norm:
            #for ddpm x_t|x_0 ~ N(alphas_bar_sqrt*x_0,1-alphas_bar)
            mean = extract(self.alphas_bar_sqrt, t, x_0).to(x_0.device) * x_0
            std = extract(self.marg_std, t, x_0).to(x_0.device)
            return mean, std
        else:
            #for ncsn x_t|x_0 ~ N(0,sqrt(beta1+...+betat))
            mean = x_0
            std = extract(self.marg_std, t, x_0).to(x_0.device)
            return mean, std

    def plot_marginal_std(self):
        '''
        plt the std of p(x_t|x_0)
        if var_norm is True,plot the relative std, else plot the absolute std
        '''
        if self.var_norm:
            plt.plot(self.marg_std/self.alphas_bar_sqrt)
        else:
            plt.plot(self.marg_std)
        plt.show()
    
    def visualize_noise(self,x0,num_display=10):

        if self.n_steps<num_display:
            num_display=self.n_steps
        
        ##uniformly choose num_display time steps from self.n_steps
        display_time_steps=torch.linspace(0,self.n_steps-1,num_display).long()

        fig, axs = plt.subplots(1, num_display, figsize=(3*num_display, 3))
        for i in range(num_display):
            q_i= self.forward_sample(x0, torch.tensor([display_time_steps[i]]))
            axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10);
            # axs[i].set_axis_off(); 
            axs[i].set_title('$q(\mathbf{x}_{'+str(display_time_steps[i].item())+'})$');



