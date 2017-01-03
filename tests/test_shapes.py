from arrows.primitive.math_arrows import *
from arrows.primitive.constant import *

from arrows.compose import compose_comb
from arrows.apply.shapes import propagate_shapes

addarr = AddArrow()
rankarr = RankArrow()

comb = compose_comb(addarr, rankarr, {0: 0})
shape = propagate_shapes(comb, [(1,2,3), (1,2,3)])
print(shape)
