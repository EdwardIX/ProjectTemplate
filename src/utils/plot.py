import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .scipyutil import compute_interpolator

def plot_heatmap(x, y, z, vmin=None, vmax=None, num=100, title='', xlabel='x', ylabel='y'):
    """
    Plot a heatmap for 3-dimensional data on the current axes.

    Parameters:
    x, y, z : array-like
        The x, y coordinates and corresponding z values.
    vmin, vmax : float, optional
        Minimum and maximum values for color scaling.
    num : int, optional
        Number of interpolation points along each dimension.
    title : str, optional
        Title of the plot.
    xlabel, ylabel : str, optional
        Labels for the x and y axes.

    Returns:
    im : matplotlib.image.AxesImage
        The image object representing the heatmap.
    """
    # Get current axes
    ax = plt.gca()

    # Create a grid for interpolation
    xx = np.linspace(np.min(x), np.max(x), num)
    yy = np.linspace(np.min(y), np.max(y), num)
    xx, yy = np.meshgrid(xx, yy)

    # Interpolate the z values
    vals = compute_interpolator(np.array([x, y]).T, np.array(z), method='cubic')(xx, yy)
    vals_0 = compute_interpolator(np.array([x, y]).T, np.array(z), method='nearest')(xx, yy)
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]

    # Reverse y coordinate for imshow compatibility
    vals = vals[::-1, :]

    # Plot heatmap
    im = ax.imshow(
        vals,
        extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
        aspect='auto',
        interpolation='bicubic',
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(im, ax=ax)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return im

def plot_lines(x, y, path=None, title='', xlabel='x', ylabel='y'):
    assert x.size == np.max(x.shape), f"Expect array x with 1 dimension, but got {x.shape}"
    x = x.reshape(-1)

    fig = plt.figure()
    if isinstance(y, dict):
        for k, v in y.items():
            assert v.size == np.max(v.shape), f"Expect array y with 1 dimension, but got {v.shape}"
            v = v.reshape(-1)
            plt.plot(x, v, label=k)
        plt.legend()
    else:
        assert y.size == np.max(y.shape), f"Expect array y with 1 dimension, but got {y.shape}"
        y = y.reshape(-1)
        plt.plot(x, y)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if path:
        plt.savefig(path)
    plt.close()
    return fig

def plot_hist(x, bins=50, path=None, xlabel='Value', ylabel='Freq', title=""):
    """
    Plot a histogram of the values in a PyTorch array.
    
    Parameters:
        array (np.array): The input array whose values are to be visualized.
        bins (int): Number of bins for the histogram (default is 50).
        title (str): Title of the histogram plot.
    """
    # Flatten the array to 1D
    x = x.flatten()
    
    # Plot the histogram
    fig = plt.figure(figsize=(8, 6))
    plt.hist(x, bins=bins, range=(x.min(), x.max()), color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if path:
        plt.savefig(path)
    plt.close()
    return fig