"""Using set-generative adversarial training for amortized random variable"""
from copy import deepcopy
from arrows.arrow import Arrow
from arrows.primitive.array_arrows import (GatherArrow, StackArrow,
  TransposeArrow)
from arrows.compositearrow import CompositeArrow
from arrows.tfarrow import TfLambdaArrow
from arrows.port_attributes import (make_in_port, make_out_port, make_error_port,
  set_port_dtype)
import tensorflow as tf


def ConcatShuffleArrow(n_inputs: int, ndims: int):
  """Concatenate n_inputs inputs and shuffle
  Arrow first n_inputs inputs are arrays to shuffle
  last input is permutation vector"""
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
    assert len(args) == 2
    xs = args[0]
    perm = args[1]
    xs = tf.split(xs, axis=1, num_or_size_splits=nsamples)

    # Only the tensor as position nsamples is authentic
    is_auth = tf.equal(perm, nsamples)

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
                              lambda: tf.constant(0.0),
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
                  sample_size: int,
                  ndims: int) -> CompositeArrow:
    """
    Arrow wihch computes loss for amortized random variable using set gan.
    Args:
        arrow: Forward function
        cond_gen: Y x Z -> X - Conditional Generators
        disc: X^n -> {0,1}^n
        sample_size: n, number of samples seen by discriminator at once
        ndims: dimensionality of dims
    Returns:
        CompositeArrow: X x Z x ... Z x RAND_PERM -> d_Loss x g_Loss x Y x ... Y
    """
    # TODO: Assumes that f has single in_port and single out_port, generalize
    c = CompositeArrow(name="%s_set_gan" % arrow.name)
    cond_gens = [deepcopy(cond_gen) for i in range(sample_size)]

    in_port = c.add_port()
    make_in_port(in_port)
    c.add_edge(in_port, arrow.in_port(0))

    # Connect f(x) to generator
    for i in range(sample_size):
      c.add_edge(arrow.out_port(0), cond_gens[i].in_port(0))

    # Connect noise input to generator second inport
    for i in range(sample_size):
      noise_in_port = c.add_port()
      make_in_port(noise_in_port)
      c.add_edge(noise_in_port, cond_gens[i].in_port(1))

    stack_shuffle = ConcatShuffleArrow(sample_size + 1, ndims)

    # Add each output from generator to shuffle set
    for i in range(sample_size):
      c.add_edge(cond_gens[i].out_port(0), stack_shuffle.in_port(i))

    # Add the posterior sample x to the shuffle set
    c.add_edge(arrow.out_port(0), stack_shuffle.in_port(sample_size))

    rand_perm_in_port = c.add_port()
    make_in_port(rand_perm_in_port)
    set_port_dtype(rand_perm_in_port, 'int32')
    c.add_edge(rand_perm_in_port, stack_shuffle.in_port(sample_size + 1))

    gan_loss_arrow = GanLossArrow(sample_size + 1)
    c.add_edge(stack_shuffle.out_port(0), disc.in_port(0))
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


    for i in range(sample_size):
      sample = c.add_port()
      make_out_port(sample)
      c.add_edge(cond_gens[0].out_port(0), sample)

    assert c.is_wired_correctly()
    return c


# def train_cgan(gan_arrow: Arrow):
# TODO

# Make neural network example
# Freeze parameters

# Make loss arrow
# Make disc stochastic
# Parameterize disc stochasticity
# Paramterize n,
