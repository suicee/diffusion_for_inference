import torch
import torch.nn as nn


import torch.nn.functional as F

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, num_classes):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(num_classes, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class score_model_cond(nn.Module):
    def __init__(self, num_classes,x_dim,x_cond_dim):
        super().__init__()
        self.lin1 = ConditionalLinear(x_dim+x_cond_dim, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        self.lin3 = ConditionalLinear(128, 128, num_classes)
        self.lin4 = nn.Linear(128, x_dim)
    
    def forward(self, x,x_cond, y):
        x = torch.cat([x,x_cond],dim=1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

class score_model(nn.Module):
    def __init__(self, num_classes, x_dim):
        super().__init__()
        self.lin1 = ConditionalLinear(x_dim, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        self.lin3 = nn.Linear(128, x_dim)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)