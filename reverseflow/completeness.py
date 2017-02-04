"""Test of different methods of maximising completeness"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensortemplates as tt
from tensortemplates import res_net
import numpy as np

# Interactive plotting
plt.ion()

def norm(tensor):
    sqr = tf.square(tensor)
    return tf.sqrt(tf.reduce_sum(sqr, reduction_indices=1))


def circle_loss(tensor):
    s = 0.5
    norms = norm(tensor)
    sdfs = tf.maximum(0.0, norms-s)
    return tf.reduce_mean(sdfs)

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
        optimizer = tf.train.MomentumOptimizer(0.001,
                                               momentum=0.1)
        update_step = optimizer.minimize(loss)
        return update_step


def max_discrepancy_loss(phi,
                         g):
    op, params = g(phi)
    theta_samples = op[0]
    mean = tf.reduce_mean(theta_samples, reduction_indices=0)
    diffs = theta_samples - mean
    sqr_diffs = tf.reduce_sum(tf.square(diffs), reduction_indices=1)
    loss = -tf.reduce_mean(sqr_diffs)
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
    loss = loss1 + loss2
    # loss = loss1
    sub_losses = [loss1, loss2]
    update_step = gen_update_step(loss)
    train_loop(loss, update_step, phi, theta_samples, sub_losses)


def train_loop(loss, update_step, phi, theta_samples, outputs, n_iterations=1000):
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(n_iterations):
        phi_samples = np.random.rand(*phi.get_shape().as_list())
        output = sess.run([loss, update_step, theta_samples] + outputs,
                          feed_dict={phi: phi_samples})
        print("Loss: ", output[0])
        print("Losses: ", output[3:])
        update_plot(output[2])


options = {'layer_width': 2,
           'nblocks': 1,
           'block_size': 1}

train(circle_loss, template_options=options)
