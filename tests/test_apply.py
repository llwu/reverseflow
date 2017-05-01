from test_arrows import all_test_arrow_gens, test_twoxyplusx
from totality_test import totality_test
from reverseflow.invert import invert
from arrows import Arrow
from arrows.apply.apply import apply, apply_backwards
from arrows.port_attributes import is_error_port
import numpy as np


def input_gen(arrow: Arrow):
    return [np.random.rand() for i in range(arrow.num_in_ports())]

def test_apply():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(apply, all_test_arrows, input_gen, test_name="apply")

def test_apply_backwards():
    orig = test_twoxyplusx()
    arrow = invert(orig)
    outputs = [np.random.randn(2, 2) for out_port in arrow.out_ports() if not is_error_port(out_port)]
    return orig, arrow, outputs, apply_backwards(arrow, outputs)

if __name__ == '__main__':
    orig, arrow, outputs, in_vals = test_apply_backwards()
    expected = apply(orig, outputs)
    inputs = [in_vals[in_port] for in_port in arrow.in_ports()]
    redo = apply(arrow, inputs)
    print(outputs)
    print(inputs)
    print(expected)
    print(redo)
    import pdb; pdb.set_trace()
