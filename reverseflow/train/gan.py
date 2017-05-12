"""Using set-generative adversarial training for amortized random variable"""
from copy import deepcopy
from arrows.arrow import Arrow
from arrows.primitive.array_arrows import (GatherArrow, StackArrow,
  TransposeArrow)
from arrows.compositearrow import CompositeArrow
from arrows.tfarrow import TfLambdaArrow
from arrows.port_attributes import *
import tensorflow as tf


def ConcatShuffleArrow(n_inputs: int, ndims: int):
  """Concatenate n_inputs inputs and shuffle
  Arrow first n_inputs inputs are arrays to shuffle
  last input is permutation vector
  Args:
    n_inputs"""
  c = CompositeArrow(name="ConcatShuffle")

  stack = StackArrow(n_inputs, axis=0)
  for i in range(n_inputs):
    in_port = c.add_port()
    make_in_port(in_port)
    c.add_edge(in_port, stack.in_port(i))

  # Permutation vector
  perm = c.add_port()
  make_in_port(perm)
  set_port_dtype(perm, 'int32')
  gather = GatherArrow()

  c.add_edge(stack.out_port(0), gather.in_port(0))
  c.add_edge(perm, gather.in_port(1))

  # Switch the first and last dimension
  # FIXME: I do this only because gather seems to affect only first dimension
  # There's a better way, probably using gather_nd
  a = [i for i in range(ndims + 1)]
  tp_perm = [a[i+1] for i in range(len(a) - 1)] + [a[0]]

  transpose = TransposeArrow(tp_perm)
  c.add_edge(gather.out_port(0), transpose.in_port(0))

  out = c.add_port()
  make_out_port(out)
  c.add_edge(transpose.out_port(0), out)
  assert c.is_wired_correctly()
  return c


def GanLossArrow(nsamples):
  def func(args, eps=1e-6):
    """Do Gan Loss in TensorFlow
    Args:
      x : (batch_size, nsamples)
      perm: (namples)
    Returns:
      (nsamples) - Generator loss (per batch)
      (nsamples) - Discriminator loss (per batch)
    """
    with tf.name_scope("ganloss"):
      assert len(args) == 2
      xs = args[0]
      perm = args[1]

      # Only the tensor as position nsamples is authentic
      is_auth = tf.equal(perm, nsamples-1)

      # xs = tf.Print(xs, [xs], message="xs", summarize=1000)
      is_auth = tf.Print(is_auth, [is_auth, perm, nsamples-1, xs[0, 0], xs[0, 1]])
      xs = tf.split(xs, axis=1, num_or_size_splits=nsamples)
      assert len(xs) == nsamples
      losses_d = []
      losses_g = []
      for i in range(nsamples):
        x = xs[i]

        def lambda_log(v):
            return lambda: tf.log(v)
        # FIXME: Redundant computation
        losses_d.append(tf.cond(is_auth[i],
                                lambda_log(x + eps),
                                lambda_log(1 - x + eps)))
        losses_g.append(tf.cond(is_auth[i],
                                lambda: tf.zeros_like(x),
                                lambda_log(x + eps)))

      loss_d = tf.concat(losses_d, axis=1)
      loss_g = tf.concat(losses_g, axis=1)
      sum_loss_d = tf.reduce_sum(loss_d, axis=1)
      sum_loss_g = tf.reduce_sum(loss_g, axis=1)
      return [sum_loss_d, sum_loss_g]

  return TfLambdaArrow(2, 2, func=func)


def set_gan_arrow(arrow: Arrow,
                  cond_gen: Arrow,
                  disc: Arrow,
                  n_fake_samples: int,
                  ndims: int,
                  x_shapes=None,
                  z_shape=None) -> CompositeArrow:
    """
    Arrow wihch computes loss for amortized random variable using set gan.
    Args:
        arrow: Forward function
        cond_gen: Y x Z -> X - Conditional Generators
        disc: X^n -> {0,1}^n
        n_fake_samples: n, number of samples seen by discriminator at once
        ndims: dimensionality of dims
    Returns:
        CompositeArrow: X x Z x ... Z x RAND_PERM -> d_Loss x g_Loss x Y x ... Y
    """
    # TODO: Assumes that f has single in_port and single out_port, generalize
    c = CompositeArrow(name="%s_set_gan" % arrow.name)
    assert cond_gen.num_in_ports() == 2, "don't handle case of more than one Y input"
    cond_gens = [deepcopy(cond_gen) for i in range(n_fake_samples)]

    # Connect x to arrow in puts
    for i in range(arrow.num_in_ports()):
      in_port = c.add_port()
      make_in_port(in_port)
      c.add_edge(in_port, arrow.in_port(i))
      if x_shapes is not None:
        set_port_shape(in_port, x_shapes[i])

    # Connect f(x) to generator
    for i in range(n_fake_samples):
      for j in range(arrow.num_out_ports()):
        c.add_edge(arrow.out_port(j), cond_gens[i].in_port(j))

    # Connect noise input to generator second inport
    for i in range(n_fake_samples):
      noise_in_port = c.add_port()
      make_in_port(noise_in_port)
      if z_shape is not None:
        set_port_shape(noise_in_port, z_shape)
      cg_noise_in_port_id = cond_gens[i].num_in_ports() - 1
      c.add_edge(noise_in_port, cond_gens[i].in_port(cg_noise_in_port_id))

    stack_shuffles = []
    rand_perm_in_port = c.add_port()
    make_in_port(rand_perm_in_port)
    set_port_shape(rand_perm_in_port, (n_fake_samples + 1,))
    set_port_dtype(rand_perm_in_port, 'int32')

    # For every output of g, i.e. x and y if f(x, y) = z
    # Stack all the Xs from the differet samples together and shuffle
    cond_gen_non_error_out_ports = cond_gen.num_out_ports() - cond_gen.num_error_ports()
    for i in range(cond_gen_non_error_out_ports):
      stack_shuffle = ConcatShuffleArrow(n_fake_samples + 1, ndims)
      stack_shuffles.append(stack_shuffle)
      # Add each output from generator to shuffle set
      for i in range(n_fake_samples):
        c.add_edge(cond_gens[i].out_port(0), stack_shuffle.in_port(i))

      # Add the posterior sample x to the shuffle set
      c.add_edge(in_port, stack_shuffle.in_port(n_fake_samples))
      c.add_edge(rand_perm_in_port, stack_shuffle.in_port(n_fake_samples + 1))

    gan_loss_arrow = GanLossArrow(n_fake_samples + 1)

    # Connect output of each stack shuffle to discriminator
    for i in range(cond_gen_non_error_out_ports):
      c.add_edge(stack_shuffles[i].out_port(0), disc.in_port(i))

    c.add_edge(disc.out_port(0), gan_loss_arrow.in_port(0))
    c.add_edge(rand_perm_in_port, gan_loss_arrow.in_port(1))

    # Add generator and discriminator loss ports
    loss_d_port = c.add_port()
    make_out_port(loss_d_port)
    make_error_port(loss_d_port)
    c.add_edge(gan_loss_arrow.out_port(0), loss_d_port)

    loss_g_port = c.add_port()
    make_out_port(loss_g_port)
    make_error_port(loss_g_port)
    c.add_edge(gan_loss_arrow.out_port(1), loss_g_port)

    # Connect fake samples to output of composition
    for i in range(n_fake_samples):
      for j in range(cond_gen_non_error_out_ports):
        sample = c.add_port()
        make_out_port(sample)
        c.add_edge(cond_gens[i].out_port(j), sample)

    # Pipe up error ports
    for cond_gen in cond_gens:
      for i in range(cond_gen_non_error_out_ports, cond_gen.num_out_ports()):
        error_port = c.add_port()
        make_out_port(error_port)
        c.add_edge(cond_gen.out_port(i), error_port)

    assert c.is_wired_correctly()
    return c

def split_ports(inv: Arrow):
  inv_in_ports = []
  inv_error_ports = []
  inv_param_ports = []
  inv_out_ports = []
  for port in inv.ports():
    if is_in_port(port):
      if is_param_port(port):
        inv_param_ports.append(port)
      else:
        inv_in_ports.append(port)
    else:
      if is_error_port(port):
        inv_error_ports.append(port)
      else:
        inv_out_ports.append(port)
  return inv_in_ports, inv_param_ports, inv_out_ports, inv_error_ports


def g_from_g_theta(inv: Arrow,
                   g_theta: Arrow):
    """
    Construct an amoritzed random variable g from a parametric inverse `inv`
    and function `g_theta` which constructs parameters for `inv`
    Args:
      g_theta: Y x Y ... Y x Z -> Theta
    Returns:
      Y x Y x .. x Z -> X x ... X x Error x ... Error
    """
    c = CompositeArrow(name="%s_g_theta" % inv.name)
    inv_in_ports, inv_param_ports, inv_out_ports, inv_error_ports = split_ports(inv)
    g_theta_in_ports, g_theta_param_ports, g_theta_out_ports, g_theta_error_ports = split_ports(g_theta)
    assert len(g_theta_out_ports) == len(inv_param_ports)

    # Connect y to g_theta and f
    for i in range(len(inv_in_ports)):
      y_in_port = c.add_port()
      make_in_port(y_in_port)
      c.add_edge(y_in_port, g_theta.in_port(i))
      c.add_edge(y_in_port, inv_in_ports[i])

    # conect up noise input to g_theta
    z_in_port = c.add_port()
    make_in_port(z_in_port)
    c.add_edge(z_in_port, g_theta.in_port(len(inv_in_ports)))

    # connect g_theta to inv
    for i in range(len(g_theta_out_ports)):
      c.add_edge(g_theta.out_port(i), inv_param_ports[i])

    for inv_out_port in inv_out_ports:
      out_port = c.add_port()
      make_out_port(out_port)
      c.add_edge(inv_out_port, out_port)

    for inv_error_port in inv_error_ports:
      error_port = c.add_port()
      make_out_port(error_port)
      make_error_port(error_port)
      c.add_edge(inv_error_port, error_port)

    assert c.is_wired_correctly()
    return c



from wacacore.train.common import (train_loop, updates, variable_summaries,
                                   setup_file_writers, get_variables)
from tensorflow import Tensor
from typing import Generator, Sequence, Callable


def train_gan_arr(sess,
                  d_loss: Tensor,
                  g_loss: Tensor,
                  train_generators: Sequence[Generator],
                  test_generators: Sequence[Generator],
                  callbacks: Sequence[Callable],
                  fetch,
                  options):
  """Train a set-generative adversarial network"""
  fetch = {} if fetch is None else fetch
  d_loss = tf.reduce_mean(- d_loss)
  g_loss = tf.reduce_mean(- g_loss)
  fetch['losses'] = {'d_loss': d_loss, 'g_loss': g_loss}

  loss_updates = []
  d_vars = get_variables('discriminator')
  loss_updates.append(updates(d_loss, d_vars, options=options)[1])
  g_vars = get_variables('generator')
  loss_updates.append(updates(g_loss, g_vars, options=options)[1])

  if 'debug' in options and options['debug'] is True:
    fetch['check'] = tf.add_check_numerics_ops()
  loss_ratios = None

  # Summaries
  # x_ten = x_tens[0]
  tf.summary.scalar("d_loss", d_loss)
  tf.summary.scalar("g_loss", g_loss)
  fetch['summaries'] = tf.summary.merge_all()
  fetch['g_loss'] = g_loss
  # tf.summary.scalar("real_variance", tf.nn.moments(x_ten, axes=[0])[1][0])
  # tf.summary.scalar("fake_variance", tf.nn.moments(fake_x_1, axes=[0])[1][0])

  # summaries = variable_summaries(losses)
  # writers = setup_file_writers('summaries', sess)
  # options['writers'] = writers

  # callbacks = [every_n(summary_writes, 25)]
  # fetch['summaries'] = summaries
  # fetch['losses'] = losses
  sess.run(tf.initialize_all_variables())
  train_loop(sess,
             loss_updates,
             fetch,
             train_generators=train_generators,
             test_generators=test_generators,
             loss_ratios=loss_ratios,
             callbacks=callbacks,
             **options)
