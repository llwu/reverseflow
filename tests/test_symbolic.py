import sympy

from arrows.apply.symbolic import symbolic_apply
from arrows.arrow import Arrow
from arrows.port import InPort

# from util import random_arrow_test
from test_arrows import test_xyplusx_flat, all_composites


def test_symbolic_apply() -> None:
    """f(x,y) = x * y + x"""
    arrow = test_xyplusx_flat()
    input_symbols = generate_input(arrow)
    output_symbols = symbolic_apply(arrow, input_symbols)

def generate_input(arrow: Arrow):
    input_symbols = []
    for i, in_port in enumerate(arrow.in_ports):
        input_symbols.append(sympy.Dummy("input_%s" % i))
    return input_symbols

def reset_and_conv(arrow: Arrow) -> None:
    input_symbols = generate_input(arrow)
    output_symbols = symbolic_apply(arrow, input_symbols)

#random_arrow_test(reset_and_conv, "to_symbolic_apply")
