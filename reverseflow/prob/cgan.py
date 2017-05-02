"""Posterior Inference with Conditional GAN"""
from arrows.config import floatX
from tensorflow import Tensor
import tensorflow as tf
from wacacore.train.common import (train_loop, updates, variable_summaries,
  setup_file_writers)
from wacacore.util.generators import infinite_samples
from wacacore.train.callbacks import every_n, summary_writes
from typing import Generator, Callable, Sequence
import numpy as np
from tflearn.layers import fully_connected

def tf_cgan(x_prior: Tensor,
            x_prior_gen: Generator,
            z: Tensor,
            z_gen: Generator,
            f: Callable,
            g: Callable,
            disc: Callable,
            options):
  """
  Train a conditional random variable using a generative adversarial network
  Args:
    prior: Prior
    prior_gen: Generator for the prior
    pcrv_inp: Poster Conditional Random Variable Inputs (placeholders)
    pcrv_inp: Poster Conditional Random Variable Outputs X
    discriminator: Tensor -> Tensor for discriminator function
  """
  eps = 1e-6
  # 1. Attach the prior to its generator
  # Construct the two loss functiosns
  y = f(x_prior)
  x_fake = g(y, z)

  # Pipe the output of cgan into discriminator
  real = disc(x_prior, y, False) + eps
  # Pipe output of prior into discriminator
  fake = disc(x_fake, y, True) - eps

  loss_d = tf.reduce_mean(-tf.log(real) - tf.log(1 - fake))
  loss_g = tf.reduce_mean(-tf.log(fake))
  losses = [loss_d, loss_g]

  # Fetch
  fetch = {'losses': losses}
  # fetch['check'] = tf.add_check_numerics_ops()
  fetch['real'] = real[0]
  fetch['fake'] = fake[0]

  # 1st element from update is update tensor, 0th is optimizer
  # Make loss updates from losses
  g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
  d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
  loss_updates = [updates(loss_d, d_vars, options)[1],
                  updates(loss_g, g_vars, options)[1]]

  loss_ratios = [1, 3]
  def generator():
    while True:
      yield {x_prior: next(x_prior_gen),
             z: next(z_gen)}

  train_generators = [generator()]

  # FIXME: Hyperparameterize this

  # Init
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  # Reconstruction loss
  y_recon = f(x_fake)
  recon_loss = tf.reduce_mean(tf.abs(y_recon - y))
  fetch['recon_loss'] = recon_loss

  # Summaries
  summaries = variable_summaries({'gen_loss': loss_g,
                                  'disc_loss': loss_d,
                                  'recon_loss': recon_loss})

  ss = tf.summary.histogram("y_recon", y_recon)
  fetch['summaries'] = summaries
  writers = setup_file_writers('summaries', sess)
  options['writers'] = writers
  callbacks = [every_n(summary_writes, 25)]

  train_loop(sess,
             loss_updates=loss_updates,
             fetch=fetch,
             train_generators=train_generators,
             test_generators=None,
             loss_ratios=loss_ratios,
             test_every=100,
             num_iterations=100000,
             callbacks=callbacks,
             **options)


def main():
  """Simple Example"""
  # x, y sampled from normal distribution
  batch_size = 512
  x_len = 2
  x_prior_gen = infinite_samples(np.random.randn,
                                 shape=(x_len,),
                                 batch_size=batch_size,
                                 add_batch=True)
  x_prior = tf.placeholder(dtype=floatX(), shape=(batch_size, x_len))

  def f(x):
    """The model"""
    return tf.reduce_sum(x, axis=1)

  z_len = 1
  z = tf.placeholder(dtype=floatX(), shape=(batch_size, z_len))
  z_gen = infinite_samples(np.random.rand,
                           shape=(z_len,),
                           batch_size=batch_size,
                           add_batch=True)

  def g(y, z):
    """Generator"""
    with tf.name_scope("generator"):
      with tf.variable_scope("generator"):
        y_exp = tf.expand_dims(y, 1)
        inp = tf.concat([y_exp, z], axis=1)
        inp = fully_connected(inp, 3, activation='elu')
        out = fully_connected(inp, x_len, activation='elu')
        return out

  def g_pi(y, z):
    """Parametric Inverse Generator"""
    with tf.name_scope("generator"):
      with tf.variable_scope("generator"):
        theta_len = 1
        # the neural network will take as input z, and output
        # the two parameters for
        inp = z
        inp = fully_connected(inp, 3, activation='elu')
        theta = fully_connected(inp, theta_len, activation='elu')
        x_1 = tf.expand_dims(y, 1) - theta
        x_2 = theta
        x = tf.concat([x_1, x_2], 1)
        return x


  def disc(x, y, reuse, use_y=True):
    """Discriminator"""
    with tf.name_scope("discriminator"):
      with tf.variable_scope("discriminator", reuse=reuse):
        if use_y:
          inp = tf.concat([x, tf.expand_dims(y, 1)], 1)
        else:
          inp = x
        # import pdb; pdb.set_trace()
        inp = fully_connected(inp, 3, activation='elu')
        out = fully_connected(inp, 1, activation='sigmoid')
        return out

  options = {'update': 'adam', 'learning_rate': 0.001, 'save': True}
  tf_cgan(x_prior,
          x_prior_gen,
          z,
          z_gen,
          f,
          g_pi,
          disc,
          options)


if __name__ == "__main__":
  main()
