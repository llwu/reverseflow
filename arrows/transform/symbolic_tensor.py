import numpy as np
from arrows.util.misc import product
import sympy

def unroll(x):
    """The point is to replace each element of indices with the corresponding theta,
    or with zero if unknown"""
    indices = np.zeros(x.indices.shape, dtype=np.dtype(object))
    for i, theta_id in np.ndenumerate(x.indices):
        if theta_id == 0:
            indices[i] = 0
        else:
            indices[i] = x.symbols[theta_id-1]
    return indices

class SymbolicTensor:
    def __init__(self, indices=None, symbols=None, shape=None, port=None, name=""):
        self.port = port
        self.name=name
        if indices is None and symbols is None and shape is not None:
            num_elements = product(shape)
            self.indices = np.arange(1, num_elements+1).reshape(shape)
            self.symbols = [sympy.Dummy("%s_theta_%s" % (name, i)) for i in range(num_elements)]
        else:
            self.indices = indices
            self.symbols = symbols

    def __eq__(self, other):
        return (self.indices == other.indices).all() and self.symbols == other.symbols

    def __ne__(self, other):
        return not self.__eq__(other)
