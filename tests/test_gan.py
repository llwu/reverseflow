"""Test GAN"""
from reverseflow.train.gan import set_gan_arrow
from arrows.std_arrows import AbsArrow, AddArrow
from arrows.apply.apply import apply
from arrows.tfarrow import TfLambdaArrow
from arrows.port_attributes import (set_port_shape, set_port_dtype)
from reverseflow.to_graph import gen_input_tensors, arrow_to_graph
import numpy as np
import tensorflow as tf
from tflearn.layers import fully_connected

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
  fwd = AbsArrow()
  n_fake_samples = 1
  n_samples = n_fake_samples + 1

  def gen_func(args):
    """Generator function"""
    with tf.name_scope("generator"):
      inp = tf.concat(args, axis=1)
      return [fully_connected(inp, 1)]

  def disc_func(args):
    """Discriminator function"""
    with tf.name_scope("discriminator"):
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

  batch_size = 16
  set_port_shape(gan_arr.in_port(0), (batch_size, 1))
  set_port_shape(gan_arr.in_port(1), (batch_size, 1))
  set_port_shape(gan_arr.in_port(2), (n_samples,))
  set_port_dtype(gan_arr.in_port(2), 'int32')

  with tf.name_scope(gan_arr.name):
      input_tensors = gen_input_tensors(gan_arr, param_port_as_var=False)
      output_tensors = arrow_to_graph(gan_arr, input_tensors)

  x_ten, z_ten, perm_ten = input_tensors
  d_loss, g_loss, fake_x_1 = output_tensors
  fetch = {'d_loss': d_loss, 'g_loss': g_loss, 'fake_x_1': fake_x_1}
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  x = np.random.rand(16, 1)
  z = np.random.rand(16, 1)
  output_data = sess.run(fetch,
                         feed_dict={x_ten: x, z_ten: z, perm_ten: perm})
  import pdb; pdb.set_trace()




test_set_gan_nn_arrow()
