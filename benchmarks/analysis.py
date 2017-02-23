"""Analyses"""
import pickle
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict

plt.ion()

def get_dirs(prefix):
    return glob.glob("%s*" % prefix)

def get_data(dirs, data_fname="last_it_3999_fetch.pickle"):
    data = []
    for path in dirs:
        fullpath = os.path.join(path, data_fname)
        data_file = open(fullpath, 'rb')
        data.append(pickle.load(data_file))
        data_file.close()
    return data

def generalization_plot(data):
    """Plot num training examples trained on vs test error"""
    # One plot for each combination of error
    error_to_data = {}
    for d in data:
        error = d[0]
        batch_size = d[1]
        if error not in error_to_data:
            error_to_data[error] = defaultdict(list)
        error_to_data[error]['x'].append(batch_size)
        error_to_data[error]['y'].append(d[3]['sub_arrow_error'])
    return error_to_data

def do_da_plotting(plot_data):
    for k, v in plot_data.items():
        X = v['x']
        Y = v['y']
        y_sort = [y for (x,y) in sorted(zip(X,Y))]
        x_sort = [x for (x,y) in sorted(zip(X,Y))]
        plt.plot(x_sort, y_sort, label=k)
    plt.legend()

prefix = "/home/zenna/data/rf/GGFDI"

# prefix
# prefix = "/home/zenna/data/rf/0BR0K"
dirs = get_dirs(prefix)
data = get_data(dirs)
options = get_data(dirs, data_fname="options.pickle")
d = [(options[i]['error'], options[i]['batch_size'], data[i]['loss'], data[i]['test_fetch_res']['loss']) for i in range(len(data))]
q = generalization_plot(d)
do_da_plotting(q)
