import numpy as np
import torch
import matplotlib.pyplot as plt



def plot_gradient(score_fn,xrange,yrange,n=20,t=0,device='cuda',ref_data=None):
    x = np.linspace(*xrange, n)
    y = np.linspace(*yrange, n)
    x, y = np.meshgrid(x, y)
    z = np.stack([x, y], axis=-1)
    z = torch.tensor(z, dtype=torch.float32).view(-1, 2).to(device)
    labels = torch.ones(z.shape[0]).long().to(device)*t
    with torch.no_grad():
        scores = score_fn(z, labels).detach().cpu().numpy()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

    plt.figure(figsize=(10,10))
    if ref_data is not None:
        plt.scatter(*ref_data.T, alpha=0.3, color='red', edgecolor='white', s=40)
    plt.quiver(*z.cpu().T, *scores_log1p.T, width=0.002, color='black')
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    plt.show()