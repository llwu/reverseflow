"""Analyses"""
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.invert import invert
from reverseflow.train.loss import inv_fwd_loss_arrow
from arrows.apply.apply import apply

import pickle
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import tensorflow as tf
import numpy as np

plt.ion()

def get_dirs(prefix):
    return glob.glob("%s*" % prefix)

def get_data(dirs, data_fname="last_it_4999_fetch.pickle"):
    data = []
    for path in dirs:
        fullpath = os.path.join(path, data_fname)
        data_file = open(fullpath, 'rb')
        data.append(pickle.load(data_file))
        data_file.close()
    return data

def get_mean(data, n_data):
    mean_data = []
    for i in range(n_data):
        t0, t1, t2, t3, t4 = data[i]
        m = 1
        for j in range(i + n_data, len(data), n_data):
            assert t0 == data[j][0]
            assert t1 == data[j][1]
            assert t4 == data[j][4]
            assert t2.keys() == data[j][2].keys()
            for key in t2.keys():
                t2[key] += data[j][2][key]
            assert t3.keys() == data[j][3].keys()
            for key in t3.keys():
                t3[key] += data[j][3][key]
            m += 1
        for key in t2.keys():
            t2[key] = t2[key]/m
        for key in t3.keys():
            t3[key] = t3[key]/m
        mean_data.append((t0, t1, t2, t3, t4))
    return mean_data

data_index = {'Data_Size': 1,
              'Test_Error': 3,
              'Training_Error': 2,
              'Num_Iterations': 4}

def generalization_plot(data, x_string, y_string):
    """Plot num training examples trained on vs test error"""
    # One plot for each combination of error
    error_to_data = {}
    for d in data:
        error = d[0]
        data_x = d[data_index[x_string]]
        data_y = d[data_index[y_string]]['inv_fwd_error']
        if error not in error_to_data:
            error_to_data[error] = defaultdict(list)
        error_to_data[error]['x'].append(data_x)
        error_to_data[error]['y'].append(data_y)
    return error_to_data

def plotting(x_string, y_string):
    # fig = plt.figure()
    plot_data_nn = main(nn_prefix, x_string, y_string)
    plot_data_pi = main(pi_prefix, x_string, y_string)
    plt.xlabel(x_string)
    plt.ylabel(y_string)
    for k, v in plot_data_nn.items():
        X = v['x']
        Y = v['y']
        y_sort = [y for (x,y) in sorted(zip(X,Y))]
        x_sort = [x for (x,y) in sorted(zip(X,Y))]
        plt.plot(x_sort, y_sort, 'g', label='nn')
    for k, v in plot_data_pi.items():
        X = v['x']
        Y = v['y']
        y_sort = [y for (x,y) in sorted(zip(X,Y))]
        x_sort = [x for (x,y) in sorted(zip(X,Y))]
        plt.plot(x_sort, y_sort, 'r', label='pi')
    plt.legend()
    plt.show(block=True)

pi_prefix = "/Users/minasyan03/Documents/urop/rf_data/rf/YPA8A"
nn_prefix = "/Users/minasyan03/Documents/urop/rf_data/rf/G1OJI"

def main(prefix, x_string, y_string):
    n_data = 10
    dirs = get_dirs(prefix)
    data = get_data(dirs)
    options = get_data(dirs, data_fname="options.pickle")
    d = [(options[i]['error'],
          options[i]['data_size'],
          data[i]['loss'],
          data[i]['test_fetch_res']['loss'],
          options[i]['num_iterations']) for i in range(len(data))]
    mean_data = get_mean(d, n_data)
    q = generalization_plot(mean_data, x_string, y_string)
    return q

def plot3d(x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_title(title)
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Loss')
    plt.show(block=True)

def parametric_plot():
    n_param = 1000
    z_value = 150

    x = tf.placeholder(tf.float32, shape=(n_param**2,))
    y = tf.placeholder(tf.float32, shape=(n_param**2,))
    # w = tf.placeholder(tf.float32, shape=(n_param**2,))
    out = x*y+x
    arrow = graph_to_arrow(output_tensors=[out],
                           input_tensors=[x, y])
    inv_arrow = invert(arrow)
    loss_arrow = inv_fwd_loss_arrow(arrow, inv_arrow)
    # import pdb; pdb.set_trace()

    z = z_value * np.ones(n_param**2)
    theta1 = np.repeat(np.linspace(1, 21, n_param), n_param)
    theta2 = np.tile(np.linspace(1, 21, n_param), n_param)
    outputs = apply(loss_arrow, inputs=[z, theta1, theta2])
    sub_arrow_loss = outputs[2]
    inv_fwd_loss = outputs[3]
    # import pdb; pdb.set_trace()

    # print(sub_arrow_loss)
    # print(inv_fwd_loss)
    # import pdb; pdb.set_trace()

    theta1 = theta1.reshape(n_param, n_param)
    theta2 = theta2.reshape(n_param, n_param)
    sub_arrow_loss = sub_arrow_loss.reshape((n_param, n_param))
    inv_fwd_loss = inv_fwd_loss.reshape((n_param, n_param))

    plot3d(theta1, theta2, sub_arrow_loss, "Surface plot for sub_arrow_error")
    plot3d(theta1, theta2, inv_fwd_loss, "Surface plot for inv_fwd_error")


if __name__ == '__main__':
    plotting('Data_Size', 'Test_Error')
    # parametric_plot()
