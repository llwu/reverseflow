"""Posterior Inference with Conditional GAN"""
from arrows import Arrow
from tensorflow import Tensor
import tensorflow as tf
from wacacore.train.common import train_loop
from typing import Generator, Callable


def tf_cgan(prior: Tensor,
            prior_gen: Generator,
            pcrv_inp: Tensor,
            pcrv_out: Tensor,
            discriminator: Callable):
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

  # Pipe the output of cgan into discriminator
  real = discriminator(prior)
  # Pipe output of prior into discriminator
  fake = discriminator(pcrv_out)

  g_min_obj = tf.log(real) + tf.log(1 - fake)
  d_min_obj = -g_min_obj
  losses = [d_min, g_min]

  # Fetch
  fetch = {'losses': losses}

  # Init
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  # Make loss updates from losses
  # TODO: FIX the appropriate parameters
  loss_updates = []

  train_generators =

  # FIXME: Hyperparameterize this
  loss_ratios = None

  train_loop(sess,
             loss_updates=loss_updates,
             fetch=fetch,
             train_generators=train_generators,
             test_generators=None,
             loss_ratios=loss_ratios,
             test_every=100,
             num_iterations=100000,
             callbacks=None,
             **kwargs)
