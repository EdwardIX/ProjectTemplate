import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .scipyutil import compute_interpolator

def plot_heatmap(x, y, z, path=None, vmin=None, vmax=None, num=100, title='', xlabel='x', ylabel='y'):
    '''
    Plot heat map for a 3-dimension data
    '''
    plt.cla()
    plt.figure()

    xx = np.linspace(np.min(x), np.max(x), num)
    yy = np.linspace(np.min(y), np.max(y), num)
    xx, yy = np.meshgrid(xx, yy)

    vals = compute_interpolator(np.array([x, y]).T, np.array(z), method='cubic')(xx, yy)
    vals_0 = compute_interpolator(np.array([x, y]).T, np.array(z), method='nearest')(xx, yy)
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]

    vals = vals[::-1, :]  # reverse y coordinate: for imshow, (0,0) show at top left.

    fig = plt.imshow(vals, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto', interpolation='bicubic', vmin=vmin, vmax=vmax)
    fig.axes.set_autoscale_on(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()

    ret = None
    if path:
        plt.savefig(path)
    else: 
        ret = plt.gcf()
    plt.close()
    return ret

def plot_lines(x, y, path=None, title='', xlabel='x', ylabel='y'):
    assert x.size == np.max(x.shape), f"Expect array x with 1 dimension, but got {x.shape}"
    x = x.reshape(-1)
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
    
    ret = None
    if path:
        plt.savefig(path)
    else: 
        ret = plt.gcf()
    plt.close()
    return ret