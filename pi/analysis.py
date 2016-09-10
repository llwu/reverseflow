import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Average over many runs
# Make it into a line plot instead of histogram


def profile2d(x,  total_time, ybins=20, max_error=None, cumulative=True):
    """
    Plot a histogram of error vs number of examples
    """
    if max_error is None:
        max_error = np.max(np.concatenate(list(x.values())))

    xbins = len(x)
    img = np.random.rand(ybins, xbins)
    for k, v in x.items():
        bincount, bin_edges = np.histogram(v, bins=ybins,
                                           range=(0.0, max_error))
        if cumulative:
            bincount = np.cumsum(bincount)
        for j, count in enumerate(bincount):
            img[j, k] = count

    # the histogram of the data
    result = plt.imshow(img, extent=[0, total_time, 0, max_error], aspect='auto')
    # l = plt.plot(bins)
    plt.ylabel('Error - |f(x*) - x|')
    plt.xlabel('Time (s)')
    plt.colorbar()
    return result, img


def profile(x, bins=20, cumulative=True, histtype='step', **kwargs):
    """
    Plot a histogram of error vs number of examples
    """
    # the histogram of the data
    result = plt.hist(x, bins=bins, cumulative=True,
                                histtype=histtype,
                                alpha=0.75, **kwargs)
    l = plt.plot(bins)
    plt.xlabel('Error - |f(x*) - x|')
    plt.ylabel('Examples per second')
    return l
