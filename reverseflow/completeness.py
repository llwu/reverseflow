"""Test of different methods of maximising completeness"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensortemplates as tt
from tensortemplates import res_net
import numpy as np

# Interactive plotting
plt.ion()
EPS = 1e-5

def norm(tensor):
    sqr = tf.square(tensor) + EPS
    return tf.sqrt(tf.reduce_sum(sqr, reduction_indices=1) + EPS) + EPS


def circle_loss(tensor):
    s = 0.5
    norms = norm(tensor) + EPS
    sdfs = tf.maximum(0.0, norms-s) + EPS
    return tf.reduce_mean(sdfs) + EPS

axis = plt.axis([-1, 1, -1, 1])
scatter = None
def update_plot(theta_samples):
    global scatter
    if scatter is None:
        scatter = plt.scatter(theta_samples[:,0], theta_samples[:,1])
    else:
        scatter.set_offsets(theta_samples)
    plt.pause(0.05)


def gen_update_step(loss):
    with tf.name_scope('optimization'):
        optimizer = tf.train.MomentumOptimizer(0.01,
                                               momentum=0.05)
        update_step = optimizer.minimize(loss)
        return update_step


def max_discrepancy_loss(phi,
                         g):
    op, params = g(phi)
    theta_samples = op[0]
    mean = tf.reduce_mean(theta_samples, reduction_indices=0) + EPS
    diffs = theta_samples - mean + EPS
    sqr_diffs = tf.reduce_sum(tf.abs(diffs), reduction_indices=1) + EPS
    loss = -tf.reduce_mean(sqr_diffs) + EPS
    return theta_samples, loss


def train(sdf,
          batch_size=512,
          phi_ndim=2,
          theta_ndim=2,
          template=res_net.template,
          template_options={}):

    phi_shape = (batch_size, phi_ndim)
    theta_shape = (batch_size, theta_ndim)
    phi = tf.placeholder(shape=phi_shape, dtype='float32',name="phi")
    def g(phi):
        return template([phi],
                        inp_shapes=[phi_shape],
                        out_shapes=[theta_shape],
                        **template_options)
    theta_samples, loss1 = max_discrepancy_loss(phi, g)
    loss2 = sdf(theta_samples)
    # loss = loss2
    lmbda = 4.0
    loss2 = lmbda*loss2
    loss = loss1 + loss2
    # loss = loss1
    variables = tf.all_variables()
    gradients1 = tf.gradients(loss1, variables)
    gradients2 = tf.gradients(loss2, variables)
    sub_losses = [loss1, loss2]
    update_step = gen_update_step(loss)
    fetches = {'loss': loss,
               'sub_losses': sub_losses,
               'update_step': update_step,
               'gradients1': gradients1,
               'gradients2': gradients2,
               'theta_samples': theta_samples}

    train_loop(loss, update_step, phi, theta_samples, fetches)

def sumsum(xs):
    return np.sum([np.sum(x) for x in xs])

def train_loop(loss, update_step, phi, theta_samples, fetches, n_iterations=1000):
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(n_iterations):
        phi_samples = np.random.rand(*phi.get_shape().as_list())
        output = sess.run(fetches,
                          feed_dict={phi: phi_samples})
        print("Loss: ", output['loss'])
        print("Losses: ", output['sub_losses'])
        print("gradients loss1: ", sumsum(output['gradients1']))
        print("gradients loss2: ", sumsum(output['gradients2']))
        update_plot(output['theta_samples'])


options = {'layer_width': 5,
           'nblocks': 1,
           'block_size': 2,
           'reuse': False}

train(circle_loss, template_options=options)
