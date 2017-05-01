from test_arrows import all_test_arrow_gens, test_twoxyplusx
from totality_test import totality_test
from reverseflow.invert import invert
from arrows import Arrow
from arrows.apply.apply import apply, apply_backwards, from_input_list
from arrows.port_attributes import is_error_port
from arrows.util.viz import show_tensorboard
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

def test_batch_apply_backwards():
    orig = test_twoxyplusx()
    inv = invert(orig)
    inputs = [[np.random.randn(2, 2) for in_port in orig.in_ports()] for i in range(10)]
    return orig, inv, from_input_list(orig, inv, inputs)

if __name__ == '__main__':
    orig, inv, the_list = test_batch_apply_backwards()
    for i in range(len(the_list)):
        inp, param, out = the_list[i]
        reconstructed_inp = apply(inv, out + param)
        print(inp)
        print(reconstructed_inp)
        print()
    import pdb; pdb.set_trace()
