"""Posterior Inference with Conditional GAN"""
import os
import sys
from arrows.config import floatX
from tensorflow import Tensor
import tensorflow as tf
from wacacore.train.common import (train_loop, updates, variable_summaries,
                                   setup_file_writers)
from wacacore.util.io import handle_args
from wacacore.util.generators import infinite_samples
from wacacore.train.callbacks import every_n, summary_writes
from wacacore.train.search import rand_local_hyper_search
from typing import Generator, Callable, Sequence
import numpy as np
from tflearn.layers import fully_connected
from tflearn.layers.normalization import batch_normalization


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

  a = -tf.log(real) - tf.log(1 - fake)
  loss_d = tf.reduce_mean(a)
  b = -tf.log(fake)
  loss_g = tf.reduce_mean(b)
  losses = [loss_d, loss_g]

  # Fetch
  fetch = {'losses': {'x_fake':x_fake[0:5],
                      'x_real':x_prior[0:5],
                      'loss_d':loss_d,
                      'loss_g':loss_g,
                      'fake':fake[0:5],
                      'real':real[0:5],
                      'd_pre_mean': a[0:5],
                      'g_pre_mean': b[0:5]}}
  # fetch['check'] = tf.add_check_numerics_ops()
  fetch['real'] = real[0]
  fetch['fake'] = fake[0]

  # 1st element from update is update tensor, 0th is optimizer
  # Make loss updates from losses
  g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
  d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                             scope='discriminator')
  loss_updates = []
  loss_updates.append(updates(loss_d, d_vars, options)[1])
  loss_updates.append(updates(loss_g, g_vars, options)[1])

  # FIXME: Hyperparameterize this
  # loss_ratios = [1, 3]
  loss_ratios = None

  def generator():
    while True:
      yield {x_prior: next(x_prior_gen),
             z: next(z_gen)}

  train_generators = [generator()]


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
             callbacks=callbacks,
             **options)


def run(options):
  """Simple Example"""
  # x, y sampled from normal distribution
  batch_size = options['batch_size']
  x_len = 1
  # x_prior_gen = infinite_samples(lambda *shape: np.random.exponential(size=shape),
  #                                shape=(x_len,),
  #                                batch_size=batch_size,
  #                                add_batch=True)
  x_prior_gen = infinite_samples(lambda *shape: np.ones(shape=shape) * 0.5,
                                 shape=(x_len,),
                                 batch_size=batch_size,
                                 add_batch=True)

  x_prior = tf.placeholder(dtype=floatX(), shape=(batch_size, x_len))

  def f(x):
    """The model"""
    # return tf.reduce_sum(x, axis=1)
    return x

  z_len = 1
  z = tf.placeholder(dtype=floatX(), shape=(batch_size, z_len))
  # z_gen = infinite_samples(np.random.rand,
  #                          shape=(z_len,),
  #                          batch_size=batch_size,
  #                          add_batch=True)
  z_gen = infinite_samples(lambda *shape: np.ones(shape=shape) * 0.5,
                                 shape=(z_len,),
                                 batch_size=batch_size,
                                 add_batch=True)


  def g(y, z):
    """Generator"""
    with tf.name_scope("generator"):
      with tf.variable_scope("generator"):
        # y = tf.expand_dims(y, 1)
        # inp = tf.concat([y, z], axis=1)
        inp = y
        inp = fully_connected(inp, 10, activation='elu')
        # inp = batch_normalization(inp)
        inp = fully_connected(inp, 10, activation='elu')
        # inp = batch_normalization(inp)
        inp = fully_connected(inp, x_len, activation='elu')
        # inp = batch_normalization(inp)
        return inp

  def g_pi(y, z):
    """Parametric Inverse Generator"""
    with tf.name_scope("generator"):
      with tf.variable_scope("generator"):
        theta_len = 1
        # the neural network will take as input z, and output
        # the two parameters for
        inp = z
        inp = fully_connected(inp, 20, activation='elu')
        inp = batch_normalization(inp)
        inp = fully_connected(inp, 20, activation='elu')
        inp = batch_normalization(inp)
        theta = fully_connected(inp, theta_len, activation='elu')
        theta = batch_normalization(theta)
        x_1 = tf.expand_dims(y, 1) - theta
        x_2 = theta
        x = tf.concat([x_1, x_2], 1)
        return x

  def disc(x, y, reuse, use_y=False):
    """Discriminator"""
    with tf.name_scope("discriminator"):
      with tf.variable_scope("discriminator", reuse=reuse):
        if use_y:
          inp = tf.concat([x, tf.expand_dims(y, 1)], 1)
        else:
          inp = x
        # import pdb; pdb.set_trace()
        # inp = fully_connected(inp, 3, activation='elu')
        out = fully_connected(inp, 1, activation='sigmoid')
        return out

  tf_cgan(x_prior,
          x_prior_gen,
          z,
          z_gen,
          f,
          g,
          disc,
          options)


def hyper_search():
  """Do hyper parameter search for cgan"""
  options = {'update': 'adam',
             'train': True,
             'save': True,
             'num_iterations': 10,
             'save_every': 1000,
             'learning_rate': 0.001,
             'batch_size': [64, 128],
             'datadir': os.path.join(os.environ['DATADIR'], "rf")}
  var_option_keys = ['batch_size']
  file_Path = os.path.abspath(__file__)
  rand_local_hyper_search(options, file_Path, var_option_keys, nsamples=2,
                    prefix='cgan', nrepeats=1)


def default_options():
    "Get default options for pdt training"
    options = {}
    options['num_iterations'] = (int, 100)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['gpu'] = (bool, False)
    options['dirname'] = (str, "dirname")
    options['datadir'] = (str, os.path.join(os.environ['DATADIR'], "rf"))
    return options


def main():
  if "--hyper" in sys.argv:
    hyper_search()
  else:
    cust_opts = default_options()
    options = handle_args(sys.argv[1:], cust_opts)
    if options['gpu']:
      print("Using GPU")
      run(options)
    else:
      print("Using CPU")
      with tf.device('/cpu:0'):
        run(options)


if __name__ == "__main__":
  main()


# TODO
# Hyper parameterize the neural network architectures
# Do hyperparameter search on openmind
# So what it should be is that I add the tag --hyper_search
