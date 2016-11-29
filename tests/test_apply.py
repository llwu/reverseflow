from reverseflow.arrows.apply.apply import apply
from test_arrows import *
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.composites.math_composites import *
import numpy as np

arr = test_xyplusx_flat()
print(apply(arr, [3.0, 4.0]))

mean = MeanArrow(2)
array1 = np.ndarray([1.0, 2.0, 3.0])
array2 = np.ndarray([1.0, 3.0, 5.0])

apply(mean, [array1, array2])
