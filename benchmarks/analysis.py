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
        t0, t1, t2, t3 = data[i]
        m = 1
        for j in range(i + n_data, len(data), n_data):
            assert t0 == data[j][0]
            assert t1 == data[j][1]
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
        mean_data.append((t0, t1, t2, t3))
    return mean_data

data_index = {'Data_Size': 1,
              'Test_Error': 3,
              'Training_Error': 2}

def generalization_plot(data, x_string, y_string):
    """Plot num training examples trained on vs test error"""
    # One plot for each combination of error
    error_to_data = {}
    for d in data:
        error = d[0]
        data_x = d[data_index[x_string]]
        data_y_1 = d[data_index[y_string[0]]]['inv_fwd_error']
        data_y_2 = d[data_index[y_string[1]]]['inv_fwd_error']
        if error not in error_to_data:
            error_to_data[error] = defaultdict(list)
        error_to_data[error]['x'].append(data_x)
        error_to_data[error]['y_train'].append(data_y_1)
        error_to_data[error]['y_test'].append(data_y_2)
    return error_to_data

def put_marker(x, y, marker):
    assert len(x) == len(y)
    marker_per_dist = 8
    for i in range(len(x) - 1):
        x_start = x[i]
        x_end = x[i + 1]
        y_start = y[i]
        y_end = y[i + 1]
        distance = np.abs(y_end-y_start) * 100 + np.abs(x_end - x_start)
        n_markers = np.floor(distance/marker_per_dist).astype(int)
        mark_x = np.linspace(x_start, x_end, n_markers)
        mark_y = np.linspace(y_start, y_end, n_markers)
        # import pdb; pdb.set_trace()

        plt.plot(mark_x, mark_y, marker)

def plotting(x_string, y_string):
    # fig = plt.figure()
    plot_data_nn = main(nn_prefix, x_string, y_string)
    plot_data_pi = main(pi_prefix, x_string, y_string)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Error')
    plt.title('Training Error vs Number of Iterations')
    for k, v in plot_data_nn.items():
        X = v['x']
        Y = v['y_train']
        y_sort = [y for (x,y) in sorted(zip(X,Y))]
        x_sort = [x for (x,y) in sorted(zip(X,Y))]
        plt.plot(x_sort, y_sort, 'g-', label='NN Train Error')
        # plt.plot(x_sort, y_sort, 'gs', label='NN Train Error')
        # put_marker(x_sort, y_sort, 'gs')
    # for k, v in plot_data_nn.items():
    #     X = v['x']
    #     Y = v['y_test']
    #     y_sort = [y for (x,y) in sorted(zip(X,Y))]
    #     x_sort = [x for (x,y) in sorted(zip(X,Y))]
    #     plt.plot(x_sort, y_sort, 'g-', label='NN Test Error')
    for k, v in plot_data_pi.items():
        X = v['x']
        Y = v['y_train']
        y_sort = [y for (x,y) in sorted(zip(X,Y))]
        x_sort = [x for (x,y) in sorted(zip(X,Y))]
        plt.plot(x_sort, y_sort, 'r--', label='PI Train Error')
        # plt.plot(x_sort, y_sort, 'r^', label='PI Train Error')
        # put_marker(x_sort, y_sort, 'r^')
    # for k, v in plot_data_pi.items():
    #     X = v['x']
    #     Y = v['y_test']
    #     y_sort = [y for (x,y) in sorted(zip(X,Y))]
    #     x_sort = [x for (x,y) in sorted(zip(X,Y))]
    #     plt.plot(x_sort, y_sort, 'r--', label='PI Test Error')

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
          data[i]['test_fetch_res']['loss']) for i in range(len(data))]
    if x_string == 'Num_Iterations':
        error_data = {}
        error = options[0]['error']
        error_data[error] = defaultdict(list)
        state = get_data(dirs, data_fname="state.pickle")

        num_iterations = len(state[0]['all_loss'][error])
        error_data[error]['x'] = np.linspace(101, num_iterations, num_iterations-100)
        error_data[error]['y_train'] = np.zeros(num_iterations-100)
        for i in range(len(state)):
            error_data[error]['y_train'] += state[i]['all_loss'][error][100:]
        error_data[error]['y_train'] = error_data[error]['y_train'] / len(state)

        return error_data
    else:
        # import pdb; pdb.set_trace()

        test_final_data = []
        train_final_data = []
        for i in range(n_data):
            test = [d[j][3]['inv_fwd_error'] for j in range(i, len(data), n_data)]
            train = [d[j][2]['inv_fwd_error'] for j in range(i, len(data), n_data)]
            test_final_data.append((np.mean(test), np.var(test)))
            train_final_data.append((np.mean(train), np.var(train)))
        # mean_test, var_test = np.mean(test_final_data), np.var(test_final_data)
        # mean_train, var_train = np.mean(train_final_data), np.var(train_final_data)
        mean_data = get_mean(d, n_data)
        import pdb; pdb.set_trace()

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
    z_value = 5

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
    theta1 = np.repeat(np.linspace(0.5, 2.5, n_param), n_param)
    theta2 = np.tile(np.linspace(1, 3, n_param), n_param)
    outputs = apply(loss_arrow, inputs=[z, theta1, theta2])
    sub_arrow_loss = outputs[2]
    inv_fwd_loss = outputs[3]

    theta1 = theta1.reshape(n_param, n_param)
    theta2 = theta2.reshape(n_param, n_param)
    sub_arrow_loss = sub_arrow_loss.reshape((n_param, n_param))
    inv_fwd_loss = inv_fwd_loss.reshape((n_param, n_param))
    # print(sub_arrow_loss)
    # print(inv_fwd_loss)
    x = np.linspace(1, 3, n_param)
    y = np.linspace(0.5, 2.5, n_param)
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Heatmap for inv_fwd_loss')
    # plt.imshow(inv_fwd_loss, extent=[5, 15, 5, 15], cmap='hot')
    plt.pcolormesh(x, y, inv_fwd_loss, cmap='hot')
    plt.colorbar()
    plt.show(block=True)

    # plot3d(theta1, theta2, sub_arrow_loss, "Surface plot for sub_arrow_error")
    # plot3d(theta1, theta2, inv_fwd_loss, "Surface plot for inv_fwd_error")


if __name__ == '__main__':
    plotting('Data_Size', ['Training_Error', 'Test_Error'])
    # parametric_plot()
