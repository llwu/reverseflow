"""Simple experiment verifying the empirical advantages of P.I. over unconstrained N.N."""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.invert import invert
from reverseflow.to_graph import arrow_to_graph, gen_input_tensors

def robot_arm(alpha, beta):
    x = tf.cos(alpha) + tf.cos(alpha+beta)
    y = tf.sin(alpha) + tf.sin(alpha+beta)
    return [x, y]

def loss_fn(x, y, alpha, beta):
    [x_, y_] = robot_arm(alpha, beta)
    return (x-x_)**2 + (y-y_)**2 + (alpha**2 + beta**2)/10

def execute_unconstrained(x, y, lr = 0.01, n_steps = 200, n_execs = 10):
    initial_losses = []
    optimal_losses = []
    for _ in range(n_execs):
        xtf = tf.constant(x)
        ytf = tf.constant(y)
        alphatf = tf.Variable(np.random.uniform(-np.pi, np.pi), dtype=tf.float64)
        betatf = tf.Variable(np.random.uniform(-np.pi, np.pi), dtype=tf.float64)
        loss = loss_fn(xtf, ytf, alphatf, betatf)
        init = tf.global_variables_initializer()
        optim_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[alphatf, betatf])

        with tf.Session() as sess:
            sess.run(init)
            for i in range(n_steps):
                _, loss_value = sess.run([optim_step, loss],
                    feed_dict={})
                if i == 0:
                    initial_losses.append(loss_value)
                if i == n_steps - 1:
                    optimal_losses.append(loss_value)
    return initial_losses, optimal_losses

def execute_pi(x, y, lr=0.01, n_steps=200, n_execs=1):
    initial_losses = []
    optimal_losses = []
    for _ in range(n_execs):
        xtf = tf.constant(x)
        ytf = tf.constant(y)
        alpha = tf.placeholder(tf.float64, shape=())
        beta = tf.placeholder(tf.float64, shape=())
        arrow = graph_to_arrow(robot_arm(alpha, beta), [alpha, beta])
        print(arrow.in_ports(), arrow.out_ports())
        print('got the arrow from tensorflow graph...')
        inv_arrow = invert(arrow)
        print(inv_arrow.in_ports(), inv_arrow.out_ports())
        print('inverted the arrow...')
        in_tensors = gen_input_tensors(inv_arrow)
        print('generated the input tensors...')
        inv_tf = arrow_to_graph(inv_arrow, in_tensors)
        print('got the tensorflow graph of the inverted arrow...')


    return initial_losses, optimal_losses



def plot_hist(initial_losses, optimal_losses, name):
    plt.hist(initial_losses, bins='auto')
    plt.title(name + "_initial")
    plt.show()
    plt.hist(optimal_losses, bins='auto')
    plt.title(name + "_optimal")
    plt.show()

if __name__ == '__main__':
    alpha = np.pi / 3
    beta = np.pi / 6
    x = np.cos(alpha) + np.cos(alpha+beta)
    y = np.sin(alpha) + np.sin(alpha+beta)
    initial_unconst, optimal_unconst = execute_pi(x, y)
    #plot_hist(initial_unconst, optimal_unconst, name='Unconstrained')
