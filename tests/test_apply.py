from reverseflow.arrows.apply.apply import apply
from test_arrows import *
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.composite.math import *
from reverseflow.arrows.compose import compose_comb_modular
import numpy as np


def test_apply():
    mean = MeanArrow(2)
    array1 = np.array([1.0, 2.0, 3.0])
    array2 = np.array([1.0, 3.0, 5.0])
    mean = apply(mean, [array1, array2])
