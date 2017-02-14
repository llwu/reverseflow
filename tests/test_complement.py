import operator
from functools import reduce

from reverseflow.util.misc import complement

def test_complement():
    shape = (3, 3, 3)
    indices = [(1, 2, 0), (0, 0, 0), (2, 2, 1), (1, 1, 1), (0, 2, 1), (0, 2, 0), (0, 1, 0)]
    output = complement(indices, shape)
    assert len(indices) + len(output) == reduce(operator.mul, list(shape), 1)
    return shape, indices, output

if __name__ == '__main__':
    shape, indices, output = test_complement()
    print("Shape: {}".format(shape))
    print("Indices: {}".format(indices))
    print("Output: {}".format(output))
