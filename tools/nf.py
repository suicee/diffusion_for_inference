import torch
from torch import nn
import normflows as nf
import numpy as np


class CondEnsembleModel(nn.Module):
    def __init__(self,models,samples_per_time_limit=10000):
        super().__init__()
        self.models=nn.ModuleList(models)
        self.samples_per_time_limit=samples_per_time_limit

        assert self.samples_per_time_limit%len(self.models)==0,'samples_per_time_limit must be divisible by number of models'
        

    def log_prob(self,x,context):
        log_prob=torch.zeros(x.shape[0],1).to(x.device)
        for model in self.models:
            log_prob+=model.log_prob(x,context)
        return log_prob/len(self.models)
    
    def _sample(self,num_samples,contexts):
        assert num_samples%len(self.models)==0,'num_samples must be divisible by number of models'
        samples=[]
        #cut context into len_model parts
        contexts=torch.chunk(contexts,len(self.models),dim=0)
        for model,context in zip(self.models,contexts):
            with torch.no_grad():
                samples.append(model.sample(num_samples//len(self.models),context)[0])
        return torch.cat(samples,dim=0)
    
    def sample(self,num_samples,contexts,to_numpy=False):
        '''
        to_numpy could realse cuda memory during sampling
        '''
        if num_samples<=self.samples_per_time_limit:
            samples=self._sample(num_samples,contexts)
            if to_numpy:
                samples=samples.detach().cpu().numpy()
            return samples
        else:
            total_samples=[]

            assert num_samples%self.samples_per_time_limit==0,"N_samples must be divisible by sample limit"

            N_chunks=num_samples//self.samples_per_time_limit
            contexts_trunks=torch.chunk(contexts,N_chunks,dim=0)
            for i in range(N_chunks):
                samples=self._sample(self.samples_per_time_limit,contexts_trunks[i])
                if to_numpy:
                    samples=samples.detach().cpu().numpy()
                total_samples.append(samples)
            if to_numpy:
                return np.concatenate(total_samples,axis=0)
            else:
                return torch.cat(total_samples,dim=0)

def get_conditional_ANSF(dim=2,context_size=2,num_layers=32):

    #need to check the meanings of these hyperparameters
    hidden_units = 128
    hidden_layers = 2

    flows = []
    for i in range(num_layers):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(dim, hidden_layers, hidden_units, 
                                                                num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(dim)]

    # Set base distribution
    q0 = nf.distributions.DiagGaussian(dim, trainable=False)
        
    # Construct flow model
    model = nf.ConditionalNormalizingFlow(q0, flows)

    return model