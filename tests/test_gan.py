"""Test GAN"""
from reverseflow.train.gan import set_gan_arrow
from arrows.std_arrows import AbsArrow, AddArrow, IdentityArrow
from arrows.apply.apply import apply
from arrows.tfarrow import TfLambdaArrow
from arrows.port_attributes import (set_port_shape, set_port_dtype)
from reverseflow.to_graph import gen_input_tensors, arrow_to_graph
import numpy as np
import tensorflow as tf
from tflearn.layers import fully_connected, batch_normalization

def test_set_gan_arrow():
  fwd = AbsArrow()
  cond_gen = AddArrow()
  disc = AbsArrow()
  gan_arr = set_gan_arrow(fwd, cond_gen, disc, 1)
  x = np.array([1.0])
  z = np.array([0.5])
  perm = np.array([1, 0])
  result = apply(gan_arr, [x, z, perm])

def test_set_gan_nn_arrow():
  fwd = IdentityArrow()
  n_fake_samples = 1
  n_samples = n_fake_samples + 1

  def gen_func(args):
    """Generator function"""
    with tf.variable_scope("generator", reuse=False):
      inp = tf.concat(args, axis=1)
      inp = fully_connected(inp, 1, activation='elu')
      inp = batch_normalization(inp)
      inp = fully_connected(inp, 1, activation='elu')
      inp = batch_normalization(inp)
      return [inp]

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
  x = np.array([1.0])
  z = np.array([0.5])
  perm = np.array([1, 0])

  batch_size = 64
  set_port_shape(gan_arr.in_port(0), (batch_size, 1))
  set_port_shape(gan_arr.in_port(1), (batch_size, 1))
  set_port_shape(gan_arr.in_port(2), (n_samples,))
  set_port_dtype(gan_arr.in_port(2), 'int32')

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
  x = np.random.rand(16, 1)
  z = np.random.rand(16, 1)
  # output_data = sess.run(fetch,
  #                        feed_dict={x_ten: x, z_ten: z, perm_ten: perm})

  from wacacore.train.common import updates, train_loop, get_variables
  losses = {'d_loss': d_loss, 'g_loss': g_loss}
  options = {'learning_rate': 0.001, 'update': 'adam'}
  d_vars = get_variables('discriminator')
  g_vars = get_variables('generator')
  loss_updates = [updates(d_loss, d_vars, options=options)[1],
                  updates(g_loss, g_vars, options=options)[1]]

  fetch['check'] = tf.add_check_numerics_ops()
  loss_ratios = [1, 1000]
  def train_gen():
    while True:
      x = np.random.rand(batch_size, 1)
      z = np.random.rand(batch_size, 1)
      perm = np.arange(n_samples)
      np.random.shuffle(perm)
      yield {x_ten: x, z_ten: z, perm_ten: perm}

  sess.run(tf.initialize_all_variables())
  train_loop(sess,
             loss_updates,
             fetch,
             train_generators=[train_gen()],
             test_generators=None,
             num_iterations=100000,
             loss_ratios=loss_ratios,
             callbacks=None)




test_set_gan_nn_arrow()
