"""GAN"""
import sys
import os
from arrows.arrow import Arrow
from reverseflow.train.gan import set_gan_arrow, g_from_g_theta
from reverseflow.to_graph import gen_input_tensors, arrow_to_graph
from arrows.std_arrows import AbsArrow, AddArrow, IdentityArrow
from arrows.tfarrow import TfLambdaArrow
import numpy as np
import tensorflow as tf
from tflearn.layers import fully_connected, batch_normalization
from wacacore.train.common import (train_loop, updates, variable_summaries,
                                   setup_file_writers, get_variables)

from wacacore.train.callbacks import every_n, summary_writes
from wacacore.train.search import rand_local_hyper_search
from wacacore.util.io import handle_args


def test_set_gan_nn_arrow(options):
  gan_arr = gan_renderer_arrow(options)
  # gan_arr = set_gan_nn_arrow(options)
  train_gan_arr(gan_arr, options)


def set_gan_nn_arrow(options):
  """Test Gan on fwd function f(x) = x"""
  fwd_arr = IdentityArrow()
  n_fake_samples = options['n_fake_samples']
  n_samples = n_fake_samples + 1
  batch_size = options['batch_size']

  def gen_func(args):
    """Generator function"""
    with tf.variable_scope("generator", reuse=False):
      # inp = tf.concat(args, axis=1)
      inp = args[0]
      inp = fully_connected(inp, 10, activation='elu')
      # inp = batch_normalization(inp)
      # inp = fully_connected(inp, 10, activation='elu')
      # inp = batch_normalization(inp)
      inp = fully_connected(inp, 1, activation='sigmoid')
      # inp = batch_normalization(inp)
      return [inp]

  # def gen_func(args):
  #   """Generator function"""
  #   with tf.variable_scope("generator", reuse=False):
  #     return [args[0]+0.2]

  def disc_func(args):
    """Discriminator function"""
    with tf.variable_scope("discriminator", reuse=False):
      assert len(args) == 1
      inp = args[0]
      inp = fully_connected(inp, 5, activation='elu')
      # inp = batch_normalization(inp)
      inp = fully_connected(inp, 5, activation='elu')
      # inp = batch_normalization(inp)
      inp = args[0]
      inp = fully_connected(inp, n_samples, activation='sigmoid')
      return [inp]

  cond_gen = TfLambdaArrow(2, 1, func=gen_func)
  disc = TfLambdaArrow(1, 1, func=disc_func)
  gan_arr = set_gan_arrow(fwd_arr, cond_gen, disc, n_fake_samples, 2,
                          x_shape=(batch_size, 1),z_shape=(batch_size, 1))

  return gan_arr


def gan_renderer_arrow(options):
  """Gan on renderer"""
  n_fake_samples = options['n_fake_samples']
  n_samples = n_fake_samples + 1
  batch_size = options['batch_size']
  res = options['res']
  width = options['width']
  height = options['height']
  nvoxels = res * res * res
  npixels = width * height

  from voxel_render import test_invert_render_graph
  fwd, inv = test_invert_render_graph(options)
  from arrows.apply.propagate import propagate
  info = propagate(inv)

  def gen_func(args):
    """Generator function"""
    with tf.variable_scope("generator", reuse=False):
      # inp = tf.concat(args, axis=1
      shapes = [info[port]['shape'] for port in inv.param_ports()]
      inp = args[0]
      inp = fully_connected(inp, 2, activation='elu')
      return [fully_connected(inp, shape[1], activation='elu') for shape in shapes]

  def disc_func(args):
    """Discriminator function"""
    with tf.variable_scope("discriminator", reuse=False):
      assert len(args) == 1
      inp = args[0]
      inp = fully_connected(inp, n_samples, activation='sigmoid')
      return [inp]

  # Make a conditional generator from the inverse\
  g_theta = TfLambdaArrow(inv.num_in_ports() - inv.num_param_ports() + 1,
                          inv.num_param_ports(), func=gen_func)
  cond_gen = g_from_g_theta(inv, g_theta)

  disc = TfLambdaArrow(1, 1, func=disc_func)
  gan_arr = set_gan_arrow(fwd, cond_gen, disc, n_fake_samples, 2,
                          x_shape=(batch_size, nvoxels), z_shape=(batch_size, 1))

  return gan_arr


def train_gan_arr(gan_arr: Arrow, options):
  n_fake_samples = options['n_fake_samples']
  n_samples = n_fake_samples + 1
  batch_size = options['batch_size']

  tf.reset_default_graph()
  with tf.name_scope(gan_arr.name):
      input_tensors = gen_input_tensors(gan_arr, param_port_as_var=False)
      output_tensors = arrow_to_graph(gan_arr, input_tensors)

  x_ten, z_ten, perm_ten = input_tensors
  d_loss, g_loss = output_tensors[0:2]
  d_loss = tf.reduce_mean(- d_loss)
  g_loss = tf.reduce_mean(- g_loss)
  # fetch = {'d_loss': d_loss, 'g_loss': g_loss, 'x_ten': x_ten, 'fake': fake_x_1}
  fetch = {'d_loss': d_loss, 'g_loss': g_loss}
  sess = tf.Session()

  losses = {'d_loss': d_loss, 'g_loss': g_loss}
  loss_updates = []
  d_vars = get_variables('discriminator')
  loss_updates.append(updates(d_loss, d_vars, options=options)[1])
  g_vars = get_variables('generator')
  loss_updates.append(updates(g_loss, g_vars, options=options)[1])

  fetch['check'] = tf.add_check_numerics_ops()
  # loss_ratios = [1, 10000]
  loss_ratios = None

  # def train_gen():
  #   """Generator for x, z and permutation"""
  #   while True:
  #     x = np.random.rand(batch_size, 1)
  #     x = np.ones(shape=(batch_size, 1)) * 0.23
  #     z = np.random.rand(batch_size, 1)
  #     # z = np.ones(shape=(batch_size, 1)) * 0.23
  #     perm = np.arange(n_samples)
  #     np.random.shuffle(perm)
  #     yield {x_ten: x, z_ten: z, perm_ten: perm}

  def train_gen():
    """Generator for x, z and permutation"""
    from wacacore.util.generators import infinite_batches
    from voxel_helpers import model_net_40
    voxel_data = model_net_40()
    x_gen = infinite_batches(voxel_data, batch_size=batch_size)
    while True:
      x = next(x_gen)
      x = x.reshape(batch_size, -1)
      z = np.random.rand(batch_size, 1)
      perm = np.arange(n_samples)
      np.random.shuffle(perm)
      yield {x_ten: x, z_ten: z, perm_ten: perm}

  # Summaries
  # import pdb; pdb.set_trace()
  tf.summary.scalar("real_variance", tf.nn.moments(x_ten, axes=[0])[1][0])
  # tf.summary.scalar("fake_variance", tf.nn.moments(fake_x_1, axes=[0])[1][0])

  summaries = variable_summaries(losses)
  writers = setup_file_writers('summaries', sess)
  options['writers'] = writers
  callbacks = [every_n(summary_writes, 25)]
  fetch['summaries'] = summaries
  fetch['losses'] = losses
  # fetch['losses']['x_fake'] = fake_x_1[0:5]
  fetch['losses']['x_real'] = x_ten[0:5]

  sess.run(tf.initialize_all_variables())
  train_loop(sess,
             loss_updates,
             fetch,
             train_generators=[train_gen()],
             test_generators=None,
             loss_ratios=loss_ratios,
             callbacks=callbacks,
             **options)



def default_render_options():
    """Default options for rendering"""
    return {'width': (int, 128),
            'height': (int, 128),
            'res': (int, 32),
            'nsteps': (int, 2),
            'nviews': (int, 1),
            'density': (float, 10.0),
            'phong': (bool, False)}


def default_options():
  "Default options for pdt training"
  options = {}
  options['num_iterations'] = (int, 100)
  options['save_every'] = (int, 100)
  options['batch_size'] = (int, 512)
  options['gpu'] = (bool, False)
  options['dirname'] = (str, "dirname")
  options['n_fake_samples'] = (int, 1)
  options['datadir'] = (str, os.path.join(os.environ['DATADIR'], "rf"))
  options.update(default_render_options())
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
