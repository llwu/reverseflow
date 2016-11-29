from reverseflow.arrows.apply.apply import apply
from test_arrows import *


arr = test_xyplusx_flat()
print(apply(arr, [3.0, 4.0]))
