import sympy
from arrows.transform.partition import partition, attachNN
from arrows.arrow import Arrow
from test_arrows import all_test_arrow_gens
from totality_test import totality_test


def test_partition():
    all_test_arrows = [gen() for gen in all_test_arrow_gens]
    totality_test(partition,
                  all_test_arrows,
                  test_name="partition")
    totality_test(attachNN,
                  all_test_arrows,
                  test_name="attachNN")

test_partition()
