import sympy
from arrows.apply.symbolic import symbolic_apply
from arrows.arrow import Arrow
from test_arrows import all_test_arrow_gens
from totality_test import totality_test
from reverseflow.train.train_y import min_approx_error_arrow

def generate_input(arrow: Arrow):
    input_symbols = []
    for i, in_port in enumerate(arrow.in_ports()):
        input_symbols.append(sympy.Dummy("input_%s" % i))
    return input_symbols

def test_train_y():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(min_approx_error_arrow,
                  all_test_arrows,
                  generate_input,
                  test_name="symbolic")
