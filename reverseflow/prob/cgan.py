"""Posterior Inference with Conditional GAN"""
from arrows.config import floatX
from tensorflow import Tensor
import tensorflow as tf
from wacacore.train.common import train_loop, updates
from wacacore.util.generators import infinite_samples
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
  # 1. Attach the prior to its generator
  # Construct the two loss functiosns
  y = f(x_prior)
  x_fake = g(y, z)

  # Pipe the output of cgan into discriminator
  real = disc(x_prior, False)
  # Pipe output of prior into discriminator
  fake = disc(x_fake, True)

  loss_d = tf.reduce_mean(-tf.log(real) + tf.log(1 - fake))
  loss_g = tf.reduce_mean(-tf.log(fake))
  losses = [loss_d, loss_g]

  # Fetch
  fetch = {'losses': losses}
  fetch['check'] = tf.add_check_numerics_ops()

  # Make loss updates from losses
  # 1st element from update is update tensor, 0th is optimizer
  g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
  d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
  import pdb; pdb.set_trace()
  loss_updates = [updates(loss_d, d_vars, options)[1],
                  updates(loss_g, g_vars, options)[1]]

  def generator():
    while True:
      yield {x_prior:next(x_prior_gen),
             z: next(z_gen)}

  train_generators = [generator()]

  # FIXME: Hyperparameterize this
  loss_ratios = None

  # Init
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  train_loop(sess,
             loss_updates=loss_updates,
             fetch=fetch,
             train_generators=train_generators,
             test_generators=None,
             loss_ratios=loss_ratios,
             test_every=100,
             num_iterations=100000,
             callbacks=None,
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

  z_len = 2
  z = tf.placeholder(dtype=floatX(), shape=(batch_size, z_len))
  z_gen = infinite_samples(np.random.randn,
                           shape=(z_len,),
                           batch_size=batch_size,
                           add_batch=True)

  def g(y, z):
    """Generator"""
    with tf.name_scope("generator"):
      with tf.variable_scope("generator"):
        y_exp = tf.expand_dims(y, 1)
        inp = tf.concat([y_exp, z], axis=1)
        out = fully_connected(inp, x_len, activation='relu')
        return out


  def disc(x, reuse):
    """Discriminator"""
    with tf.name_scope("discriminator"):
      with tf.variable_scope("discriminator", reuse=reuse):
        inp = x
        out = fully_connected(inp, 1, activation='sigmoid')
        return out

  options = {'update': 'adam', 'learning_rate': 0.005}
  tf_cgan(x_prior,
          x_prior_gen,
          z,
          z_gen,
          f,
          g,
          disc,
          options)


if __name__ == "__main__":
  main()
