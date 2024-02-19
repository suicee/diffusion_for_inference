import torch
from matplotlib import pyplot as plt



def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class noise_scheduler():
    def __init__(self, sigma_start, sigma_end, n_steps,var_norm=True,schedule='linear'):
        '''
        var norm is the flag to use the variance normalization or not(corresponding to DDPM and NCSN)
        however note that for current implementation, sigma have different meaning for var_norm=True and var_norm=False
        for var_norm=True, sigma is the std of the conditional distribution of every step
        for var_norm=False, sigma is the std of the marginal distribution of every step
        this should be unified in the future
        '''
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.n_steps = n_steps
        self.schedule = schedule
        self.var_norm = var_norm

        betas=self.make_beta_schedule(schedule=self.schedule,
                                             n_timesteps=self.n_steps, 
                                             start=self.sigma_start, 
                                             end=self.sigma_end)

        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        self.sigmas=torch.sqrt(betas)
        self.alphas=alphas
        self.alphas_prod=alphas_prod
        self.alphas_bar_sqrt=alphas_bar_sqrt
        self.one_minus_alphas_bar_log=one_minus_alphas_bar_log
        self.one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt

    @staticmethod
    def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    
    def forward_sample(self,x_0, t):
        noise = torch.randn_like(x_0).to(x_0.device)
        with torch.no_grad():
            if self.var_norm:
                alphas_t = extract(self.alphas_bar_sqrt, t, x_0).to(x_0.device)
                alphas_1_m_t = extract(self.one_minus_alphas_bar_sqrt, t, x_0).to(x_0.device)
                return alphas_t * x_0 + alphas_1_m_t * noise,noise
            else:
                used_sigmas=extract(self.sigmas,t,x_0).to(x_0.device)
                return x_0+used_sigmas*noise,used_sigmas

    def plot_sigmas(self):
        plt.plot(self.sigmas)
        plt.show()
    
    def visualize_noise(self,x0,num_display=10):

        if self.n_steps<num_display:
            num_display=self.n_steps

        ##uniformly choose num_display time steps from self.n_steps
        display_time_steps=torch.linspace(0,self.n_steps-1,num_display).long()

        fig, axs = plt.subplots(1, num_display, figsize=(3*num_display, 3))
        for i in range(num_display):
            q_i,_ = self.forward_sample(x0, torch.tensor([display_time_steps[i]]))
            axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10);
            axs[i].set_axis_off(); axs[i].set_title('$q(\mathbf{x}_{'+str(display_time_steps[i].item())+'})$')



