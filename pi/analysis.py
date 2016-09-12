import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Average over many runs
# Make it into a line plot instead of histogram

def plot():
    std_loss_hists, domain_loss_hists, total_times = x

    import matplotlib.pyplot as plt
    import pi
    for k, v in std_loss_hists.items():
        if k == 'nnet': continue
        print(k)
        print(k=='nnet')
        pi.analysis.profile2d(v, total_times[k], max_error=50)
        plt.title('std_loss %s' % k)
        plt.figure()

    for k, v in domain_loss_hists.items():
        print(k)
        if k == 'nnet': continue
        pi.analysis.profile2d(v, total_times[k], max_error=50)
        plt.title('domain_loss %s' % k)
        plt.figure()

def cumfreq(a, numbins=10, defaultreallimits=None):
    # docstring omitted
    h,l,b,e = np.histogram(a,numbins,defaultreallimits)
    cumhist = np.cumsum(h*1, axis=0)
    return cumhist,l,b,e

def plot_cdf(data, num_bins):
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    cdf = np.cumsum(counts)
    print(cdf)
    plt.plot(bin_edges[1:], cdf)


def sort_plot(data):
    data = np.sort(data)
    plt.plot(data, np.arange(len(data))/len(data))

def plot_cdfs(loss_hist, t, num_bins=100):
    legend = []
    for k, v in loss_hist.items():
        data = loss_hist[k][t]
        data = np.sort(data)
        plt.semilogx(data, np.arange(len(data))/len(data))
        legend.append(k)

    plt.legend(legend, loc='upper left')
    plt.ylabel('Count')
    plt.xlabel('Domain Error - f(x*)')




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
