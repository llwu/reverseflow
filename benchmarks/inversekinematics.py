import sys
import getopt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arrows.port_attributes import has_port_label
from reverseflow.util.tf import *
from arrows.util.viz import show_tensorboard_graph
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from reverseflow.train.loss import inv_fwd_loss_arrow


plt.ion()
c = tf.cos
s = tf.sin


def ik_fwd_f(inputs):
    phi1 = inputs[0]
    phi2 = inputs[1]
    phi4 = inputs[2]
    phi5 = inputs[3]
    phi6 = inputs[4]
    d2 = inputs[5]
    d3 = 2.0
    h1 = 1.0
    r11 = -s(phi6)*(c(phi4)*s(phi1) + c(phi1)*c(phi2)*s(phi4) ) - c(phi6)*(c(phi5)*(s(phi1)*s(phi4) - c(phi1)*c(phi2)*c(phi4) ) + c(phi1)*s(phi2)*s(phi5) )
    r12 = s(phi6)*(c(phi5)*(s(phi1)*s(phi4) - c(phi1)*c(phi2)*c(phi4) ) + c(phi1)*s(phi2)*s(phi5) ) - c(phi6)*(c(phi4)*s(phi1) + c(phi1)*c(phi2)*s(phi4) )
    r13 = s(phi5)*(s(phi1)*s(phi4) - c(phi1)*c(phi2)*c(phi4) ) - c(phi1)*c(phi5)*s(phi2)
    r21 = s(phi6)*(c(phi1)*c(phi4) - c(phi2)*s(phi1)*s(phi4) ) + c(phi6)*(c(phi5)*(c(phi1)*s(phi4) + c(phi2)*c(phi4)*s(phi1) ) - s(phi1)*s(phi2)*s(phi5) )
    r22 = c(phi6)*(c(phi1)*c(phi4) - c(phi2)*s(phi1)*s(phi4) ) - s(phi6)*(c(phi5)*(c(phi1)*s(phi4) + c(phi2)*c(phi4)*s(phi1) ) - s(phi1)*s(phi2)*s(phi5) )
    r23 = -s(phi5)*(c(phi1)*s(phi4) + c(phi2)*c(phi4)*s(phi1) ) - c(phi5)*s(phi1)*s(phi2)
    r31 = c(phi6)*(c(phi2)*s(phi5) + c(phi4)*c(phi5)*s(phi2) ) - s(phi2)*s(phi4)*s(phi6)
    r32 = -s(phi6)*(c(phi2)*s(phi5) + c(phi4)*c(phi5)*s(phi2) ) - c(phi6)*s(phi2)*s(phi4)
    r33 = c(phi2)*c(phi5) - c(phi4)*s(phi2)*s(phi5)
    px = d2*s(phi1) - d3*c(phi1)*s(phi2)
    py = -d2*c(phi1) - d3*s(phi1)*s(phi2)
    pz = h1 + d3*c(phi2)
    outputs = [px, py, pz]
    outputs.extend([r11,
                    r12,
                    r13,
                    r21,
                    r22,
                    r23,
                    r31,
                    r32,
                    r33])
    return outputs

fig = plt.figure()

def plot_robot_arm(inputs, target):
    global fig
    phi1 = inputs[0]
    phi2 = inputs[1]
    phi4 = inputs[2]
    phi5 = inputs[3]
    phi6 = inputs[4]
    d2 = inputs[5]
    d3 = 2.0
    h1 = 1.0
    T = []      # list of T matrices
    T.append(np.array([[np.cos(phi1), -np.sin(phi1), 0, 0],
                       [np.sin(phi1), np.cos(phi1), 0, 0],
                       [0, 0, 1, h1],
                       [0, 0, 0, 1]]))
    T.append(np.array([[np.cos(phi2), -np.sin(phi2), 0, 0],
                       [0, 0, -1, -d2],
                       [np.sin(phi2), np.cos(phi2), 0, 0],
                       [0, 0, 0, 1]]))
    T.append(np.array([[1, 0, 0, 0],
                       [0, 0, 1, d3],
                       [0, -1, 0, 0],
                       [0, 0, 0, 1]]))
    T.append(np.array([[np.cos(phi4), -np.sin(phi4), 0, 0],
                       [np.sin(phi4), np.cos(phi4), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]))
    T.append(np.array([[np.cos(phi5), -np.sin(phi5), 0, 0],
                       [0, 0, -1, 0],
                       [np.sin(phi5), np.cos(phi5), 0, 0],
                       [0, 0, 0, 1]]))
    T.append(np.array([[np.cos(phi6), -np.sin(phi6), 0, 0],
                       [0, 0, -1, 0],
                       [-np.sin(phi6), -np.cos(phi6), 0, 0],
                       [0, 0, 0, 1]]))

    startX = 0
    startY = 0
    startZ = 0
    startT = np.identity(4)
    ax = fig.gca(projection='3d')
    scatter_x = []
    scatter_y = []
    scatter_z = []
    for i in range(6):
        scatter_x.append(startX)
        scatter_y.append(startY)
        scatter_z.append(startZ)
        newT = np.matmul(startT, T[i])
        newX = newT[0, 3]
        newY = newT[1, 3]
        newZ = newT[2, 3]
        # print(newT)
        plot_3dline(startX, startY, startZ, newX, newY, newZ, ax)
        startT = newT
        startX = newX
        startY = newY
        startZ = newZ
    scatter_x.append(startX)
    scatter_y.append(startY)
    scatter_z.append(startZ)
    ax.scatter(scatter_x, scatter_y, scatter_z, c='b')
    ax.scatter(target[0], target[1], target[2], c='r')
    ax.legend()
    plt.show()

def plot_3dline(x1, y1, z1, x2, y2, z2, ax):
    x = np.linspace(x1, x2, 100)
    y = np.linspace(y1, y2, 100)
    z = np.linspace(z1, z2, 100)
    ax.plot(x, y, z, label='robot_arm')


def ik_gen_graph(g, batch_size, is_placeholder):
    with g.name_scope("fwd_g"):
        inputs = [0]*6
        inputs[0] = tf.placeholder(tf.float32, name="phi1", shape=())
        inputs[1] = tf.placeholder(tf.float32, name="phi2", shape=())
        # inputs['phi3'] = placeholder(tf.float32, name="phi3", shape=(batch_size, 1))
        inputs[2] = tf.placeholder(tf.float32, name="phi4", shape=())
        inputs[3] = tf.placeholder(tf.float32, name="phi5", shape=())
        inputs[4] = tf.placeholder(tf.float32, name="phi6", shape=())

        inputs[5] = tf.placeholder(tf.float32, name="d2", shape=())
        # inputs[6] = tf.placeholder(tf.float32, name="d3", shape=())
        # inputs[7] = tf.Variable(1.0, name="h1")

        outputs = ik_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}


def test_ik():
    with tf.name_scope("ik_stanford_manipulator"):
        in_out = ik_gen_graph(tf.Graph(), 1, is_placeholder)
        input_values = [30, 45, 90, 0, 60, 1]
        with tf.Session() as sess:
            output_values = sess.run(in_out["outputs"], feed_dict={in_out["inputs"][i]: input_values[i] for i in range(len(input_values))})
        print(output_values)
        # plot_robot_arm(input_values, (output_values[0], output_values[1], output_values[2]))
    arrow = graph_to_arrow(output_tensors=in_out["outputs"],
                           input_tensors=in_out["inputs"],
                           name="ik_stanford")
    # show_tensorboard_graph()
    tf.reset_default_graph()
    inverse = invert(arrow)
    inv_fwd_arrow = inv_fwd_loss_arrow(arrow)
    target = (output_values[0], output_values[1], output_values[2])
    def plot_call_back(fetch_res, target=target):
        robot_joints = fetch_res['output_tensors'][0:6]
        r = np.array(robot_joints).flatten()
        fig.clear()
        plot_robot_arm(list(r), target)
        plt.pause(0.01)

    min_approx_error_arrow(inv_fwd_arrow,
                           output_values,
                           error_filter=lambda port: has_port_label(port, "inv_fwd_error"),
                           output_call_back=plot_call_back)

test_ik()
