import numpy as np
import matplotlib.pyplot as plt

#codes modified from ltu-ili for evaluating sbi results
def get_ranks(
    samples: np.array,
    trues: np.array,
) -> np.array:
    """Get the marginal ranks of the true parameters in the posterior samples.

    Args:
        samples (np.array): posterior samples of shape (nsamples, ndata, npars)
        trues (np.array): true parameters of shape (ndata, npars)

    Returns:
        np.array: ranks of the true parameters in the posterior samples 
            of shape (ndata, npars)
    """
    ranks = (samples < trues[None, ...]).sum(axis=0)
    return ranks

def plot_ranks_histogram(
    samples: np.ndarray, trues: np.ndarray,
     nbins: int = 10
) -> plt.Figure:
    """
    Plot a histogram of ranks for each parameter.

    Args:
        samples (numpy.ndarray): List of samples.
        trues (numpy.ndarray): Array of true values.
        signature (str): Signature for the histogram file name.
        nbins (int, optional): Number of bins for the histogram. Defaults to 10.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    """
    ndata, npars = trues.shape
    navg = ndata / nbins
    ranks = get_ranks(samples, trues)

    fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))
    if npars == 1:
        ax = [ax]

    for i in range(npars):
        ax[i].hist(np.array(ranks)[:, i], bins=nbins)
        # ax[i].set_title(self.labels[i])
    ax[0].set_ylabel('counts')

    for axis in ax:
        axis.set_xlim(0, ranks.max())
        axis.set_xlabel('rank')
        axis.grid(visible=True)
        axis.axhline(navg, color='k')
        axis.axhline(navg - navg ** 0.5, color='k', ls="--")
        axis.axhline(navg + navg ** 0.5, color='k', ls="--")

    return fig

def plot_coverage(
    samples: np.ndarray, trues: np.ndarray,
    plotscatter: bool = True
) -> plt.Figure:
    """
    Plot the coverage of predicted percentiles against empirical percentiles.

    Args:
        samples (numpy.ndarray): Array of predicted samples.
        trues (numpy.ndarray): Array of true values.
        signature (str): Signature for the plot file name.
        plotscatter (bool, optional): Whether to plot the scatter plot. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    """
    ndata, npars = trues.shape
    ranks = get_ranks(samples, trues)

    unicov = [np.sort(np.random.uniform(0, 1, ndata)) for j in range(200)]
    unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

    fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
    if npars == 1:
        ax = [ax]
    cdf = np.linspace(0, 1, len(ranks))
    for i in range(npars):
        xr = np.sort(ranks[:, i])
        xr = xr / xr[-1]
        ax[i].plot(cdf, cdf, 'k--')
        if plotscatter:
            ax[i].fill_between(cdf, unip[0], unip[-1],
                                color='gray', alpha=0.2)
            ax[i].fill_between(cdf, unip[1], unip[-2],
                                color='gray', alpha=0.4)
        ax[i].plot(xr, cdf, lw=2, label='posterior')
        ax[i].set(adjustable='box', aspect='equal')
        # ax[i].set_title(self.labels[i])
        ax[i].set_xlabel('Predicted Percentile')
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)

    ax[0].set_ylabel('Empirical Percentile')
    for axis in ax:
        axis.grid(visible=True)

    return fig

def plot_predictions(
    samples: np.ndarray, trues: np.ndarray,
) -> plt.Figure:
    """
    Plot the mean and standard deviation of the predicted samples against
    the true values.

    Args:
        samples (np.ndarray): Array of predicted samples.
        trues (np.ndarray): Array of true values.
        signature (str): Signature for the plot.

    Returns:
        plt.Figure: The plotted figure.
    """
    npars = trues.shape[-1]
    mus, stds = samples.mean(axis=0), samples.std(axis=0)

    fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
    if npars == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for j in range(npars):
        axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                        fmt="none", elinewidth=0.5, alpha=0.5)
        axs[j].plot(
            *(2 * [np.linspace(min(trues[:, j]), max(trues[:, j]), 10)]),
            'k--', ms=0.2, lw=0.5)
        axs[j].grid(which='both', lw=0.5)
        axs[j].set(adjustable='box', aspect='equal')
        # axs[j].set_title(self.labels[j], fontsize=12)
        axs[j].set_xlabel('True')
    axs[0].set_ylabel('Predicted')

    return fig