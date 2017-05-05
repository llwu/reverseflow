"""Using set-generative adversarial training for amortized random variable"""
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from arrows.tfarrow import TfArrow
from arrows.port_attributes import make_in_port, make_out_port, make_error_port
from copy import deepcopy


def ConcatShuffleArrow(n):
  return TfArrow(n + 1, 1)


def GanLossArrow():
  return TfArrow(2, 1)


def set_gan_arrow(arrow: Arrow,
                  cond_gen: Arrow,
                  disc: Arrow,
                  sample_size: int) -> CompositeArrow:
    """
    Arrow wihch computes loss for amortized random variable using set gan.
    Args:
        arrow: Forward function
        cond_gen: Y x Z -> X - Conditional Generators
        disc: X^n -> {0,1}^n
        sample_size: n, number of samples seen by discriminator at once
    Returns:
        CompositeArrow: X x Z x ... Z x RAND_PERM -> Loss x Y x ... Y
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

    concat_shuffle = ConcatShuffleArrow(sample_size)
    for i in range(sample_size):
      c.add_edge(cond_gens[i].out_port(0), concat_shuffle.in_port(i))

    rand_perm_in_port = c.add_port()
    make_in_port(rand_perm_in_port)
    c.add_edge(rand_perm_in_port, concat_shuffle.in_port(sample_size))

    gan_loss_arrow = GanLossArrow()
    c.add_edge(concat_shuffle.out_port(0), disc.in_port(0))
    c.add_edge(disc.out_port(0), gan_loss_arrow.in_port(0))
    c.add_edge(rand_perm_in_port, gan_loss_arrow.in_port(1))

    loss_port = c.add_port()
    make_out_port(loss_port)
    make_error_port(loss_port)
    c.add_edge(gan_loss_arrow.out_port(0), loss_port)

    for i in range(sample_size):
      sample = c.add_port()
      make_out_port(sample)
      c.add_edge(cond_gens[0].out_port(0), sample)

    assert c.is_wired_correctly()
    return c
