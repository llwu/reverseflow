"""GAN"""
import sys
import os
from reverseflow.train.gan import set_gan_arrow
from reverseflow.to_graph import gen_input_tensors, arrow_to_graph
from arrows.arrow import Arrow
from arrows.std_arrows import AbsArrow, AddArrow, IdentityArrow
from arrows.tfarrow import TfLambdaArrow
from arrows.port_attributes import (set_port_shape, set_port_dtype)
import numpy as np
import tensorflow as tf
from tflearn.layers import fully_connected, batch_normalization
from wacacore.train.common import (train_loop, updates, variable_summaries,
                                   setup_file_writers, get_variables)

from wacacore.train.callbacks import every_n, summary_writes
from wacacore.train.search import rand_local_hyper_search
from wacacore.util.io import handle_args


def test_set_gan_nn_arrow(options):
  gan_arr = set_gan_nn_arrow(options)
  train_gan_arr(gan_arr, options)


def set_gan_nn_arrow(options):
  fwd = IdentityArrow()
  n_fake_samples = options['n_fake_samples']
  n_samples = n_fake_samples + 1
  batch_size = options['batch_size']

  def gen_func(args):
    """Generator function"""
    with tf.variable_scope("generator", reuse=False):
      inp = tf.concat(args, axis=1)
      inp = fully_connected(inp, 1, activation='elu')
      inp = batch_normalization(inp)
      inp = fully_connected(inp, 1, activation='elu')
      inp = batch_normalization(inp)
      return [inp]

  # def gen_func(args):
  #   """Generator function"""
  #   with tf.variable_scope("generator", reuse=False):
  #     return [args[0]]

  def disc_func(args):
    """Discriminator function"""
    with tf.variable_scope("discriminator", reuse=False):
      assert len(args) == 1
      inp = args[0]
      l1 = fully_connected(inp, n_samples, activation='sigmoid')
      return [l1]

  cond_gen = TfLambdaArrow(2, 1, func=gen_func)
  disc = TfLambdaArrow(1, 1, func=disc_func)
  gan_arr = set_gan_arrow(fwd, cond_gen, disc, n_fake_samples, 2)


  set_port_shape(gan_arr.in_port(0), (batch_size, 1))
  set_port_shape(gan_arr.in_port(1), (batch_size, 1))
  set_port_shape(gan_arr.in_port(2), (n_samples,))
  set_port_dtype(gan_arr.in_port(2), 'int32')
  return gan_arr


def train_gan_arr(gan_arr: Arrow, options):
  n_fake_samples = options['n_fake_samples']
  n_samples = n_fake_samples + 1
  batch_size = options['batch_size']

  with tf.name_scope(gan_arr.name):
      input_tensors = gen_input_tensors(gan_arr, param_port_as_var=False)
      output_tensors = arrow_to_graph(gan_arr, input_tensors)

  x_ten, z_ten, perm_ten = input_tensors
  d_loss, g_loss, fake_x_1 = output_tensors
  d_loss = tf.reduce_mean(- d_loss)
  g_loss = tf.reduce_mean(- g_loss)
  # fetch = {'d_loss': d_loss, 'g_loss': g_loss, 'x_ten': x_ten, 'fake': fake_x_1}
  fetch = {'d_loss': d_loss, 'g_loss': g_loss}
  sess = tf.Session()
  # output_data = sess.run(fetch,
  #                        feed_dict={x_ten: x, z_ten: z, perm_ten: perm})

  losses = {'d_loss': d_loss, 'g_loss': g_loss}
  loss_updates = []
  d_vars = get_variables('discriminator')
  loss_updates.append(updates(d_loss, d_vars, options=options)[1])
  g_vars = get_variables('generator')
  loss_updates.append(updates(d_loss, g_vars, options=options)[1])

  fetch['check'] = tf.add_check_numerics_ops()
  # loss_ratios = [1, 1]
  loss_ratios = None

  def train_gen():
    """Generator for x, z and permutation"""
    while True:
      x = np.random.rand(batch_size, 1)
      z = np.random.rand(batch_size, 1)
      perm = np.arange(n_samples)
      np.random.shuffle(perm)
      yield {x_ten: x, z_ten: z, perm_ten: perm}

  # Summaries
  summaries = variable_summaries(losses)
  writers = setup_file_writers('summaries', sess)
  options['writers'] = writers
  callbacks = [every_n(summary_writes, 25)]
  fetch['summaries'] = summaries
  fetch['losses'] = losses

  sess.run(tf.initialize_all_variables())
  train_loop(sess,
             loss_updates,
             fetch,
             train_generators=[train_gen()],
             test_generators=None,
             loss_ratios=loss_ratios,
             callbacks=callbacks,
             **options)


def default_options():
  "Get default options for pdt training"
  options = {}
  options['num_iterations'] = (int, 100)
  options['save_every'] = (int, 100)
  options['batch_size'] = (int, 512)
  options['gpu'] = (bool, False)
  options['dirname'] = (str, "dirname")
  options['n_fake_samples'] = (int, 1)
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
      test_set_gan_nn_arrow(options)
    else:
      print("Using CPU")
      with tf.device('/cpu:0'):
        test_set_gan_nn_arrow(options)


if __name__ == '__main__':
  main()
