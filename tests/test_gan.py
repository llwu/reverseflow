"""Test GAN"""
from reverseflow.train.gan import set_gan_arrow
from arrows.std_arrows import AbsArrow, AddArrow


def test_set_gan_arrow():
  fwd = AbsArrow()
  cond_gen = AddArrow()
  disc = AbsArrow()
  set_gan_arrow(fwd, cond_gen, disc, 3)

test_set_gan_arrow()
