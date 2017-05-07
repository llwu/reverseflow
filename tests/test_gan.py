"""Test GAN"""
from reverseflow.train.gan import set_gan_arrow
from arrows.std_arrows import AbsArrow, AddArrow
from arrows.apply.apply import apply
import numpy as np

def test_set_gan_arrow():
  fwd = AbsArrow()
  cond_gen = AddArrow()
  disc = AbsArrow()
  gan_arr = set_gan_arrow(fwd, cond_gen, disc, 1)
  x = np.array([1.0])
  z = np.array([0.5])
  perm = np.array([1, 0])
  result = apply(gan_arr, [x, z, perm])
  import pdb; pdb.set_trace()

test_set_gan_arrow()
