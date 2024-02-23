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
    
# class score_model_cond(nn.Module):
#     def __init__(self, num_classes,x_dim,x_cond_dim):
#         super().__init__()
#         self.lin1 = ConditionalLinear(x_dim+x_cond_dim, 128, num_classes)
#         self.lin2 = ConditionalLinear(128, 128, num_classes)
#         self.lin3 = ConditionalLinear(128, 128, num_classes)
#         self.lin4 = nn.Linear(128, x_dim)
    
#     def forward(self, x,x_cond, y):
#         x = torch.cat([x,x_cond],dim=1)
#         x = F.softplus(self.lin1(x, y))
#         x = F.softplus(self.lin2(x, y))
#         x = F.softplus(self.lin3(x, y))
#         return self.lin4(x)

# class score_model(nn.Module):
#     def __init__(self, num_classes, x_dim):
#         super().__init__()
#         self.lin1 = ConditionalLinear(x_dim, 128, num_classes)
#         self.lin2 = ConditionalLinear(128, 128, num_classes)
#         self.lin3 = nn.Linear(128, x_dim)
    
#     def forward(self, x, y):
#         x = F.softplus(self.lin1(x, y))
#         x = F.softplus(self.lin2(x, y))
#         return self.lin3(x)

class score_model_mlp_with_condlinear(nn.Module):
    def __init__(self, x_dim, n_steps, hidden_dim=[128,128]):
        '''
        mlp score model1
        activation function: softplus
        dropout: None
        embedding: use conditional linear layer
        '''
        super().__init__()
        self.input = ConditionalLinear(x_dim, hidden_dim[0], n_steps)
        self.hidden = nn.ModuleList([ConditionalLinear(hidden_dim[i], hidden_dim[i+1], n_steps) for i in range(len(hidden_dim)-1)])
        self.output = nn.Linear(hidden_dim[-1], x_dim)
    
    def forward(self, x, t):
        x = F.softplus(self.input(x, t))
        for layer in self.hidden:
            x = F.softplus(layer(x, t))
        return self.output(x)

class score_model_mlp_cond_with_condlinear(nn.Module):
    def __init__(self, x_dim, x_cond_dim, n_steps, hidden_dim=[128,128]):
        super().__init__()
        self.input = ConditionalLinear(x_dim+x_cond_dim, hidden_dim[0], n_steps)
        self.hidden = nn.ModuleList([ConditionalLinear(hidden_dim[i], hidden_dim[i+1], n_steps) for i in range(len(hidden_dim)-1)])
        self.output = nn.Linear(hidden_dim[-1], x_dim)
    
    def forward(self, x, x_cond, t):
        x = torch.cat([x,x_cond],dim=1)
        x = F.softplus(self.input(x, t))
        for layer in self.hidden:
            x = F.softplus(layer(x, t))
        return self.output(x)
    

class score_model_mlp(nn.Module):
    def __init__(self,input_dim,n_steps,hidden=[128,128]):
        '''
        current model
        activation function: GELU
        dropout: 0.2
        embedding: directly add to input
        '''
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.Dropout(0.2),
            nn.GELU()
        )
        # Condition time t
        self.embedding_layer = nn.Embedding(n_steps, hidden[0])
        
        self.hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden[i], hidden[i+1]),
            nn.Dropout(0.2),
            nn.GELU()
        ) for i in range(len(hidden)-1)])

        self.output = nn.Linear(hidden[-1], input_dim)

    def forward(self, x, t):
        x = self.input(x) + self.embedding_layer(t)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x
    
class score_model_mlp_cond():
    def __init__(self, input_dim, cond_dim, n_steps, hidden=[128,128]):
        '''
        current model
        activation function: GELU
        dropout: 0.2
        embedding: directly add to input
        '''
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim+cond_dim, hidden[0]),
            nn.Dropout(0.2),
            nn.GELU()
        )
        # Condition time t
        self.embedding_layer = nn.Embedding(n_steps, hidden[0])
        
        self.hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden[i], hidden[i+1]),
            nn.Dropout(0.2),
            nn.GELU()
        ) for i in range(len(hidden)-1)])

        self.output = nn.Linear(hidden[-1], input_dim)

    def forward(self, x, x_cond, t):
        x = torch.cat([x,x_cond],dim=1)
        x = self.input(x) + self.embedding_layer(t)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x