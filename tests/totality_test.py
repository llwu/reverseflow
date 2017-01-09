from arrows import Arrow
from typing import Sequence, Callable

def totality_test(f:Callable,
                  arrows: Sequence[Arrow],
                  input_gen:Callable=None,
                  test_name:str=None):
    """Helper to check that arrow simply runs.
    args:
        f: a function which takes an arrow as its first input
        arrow: An arrow to test
        input_gen: function which generates input from arrow, if None
                   then no input is supplied to testing f
    """
    print("Testing %s" % test_name)
    for i, arrow in enumerate(arrows):
        print("Arrow %s of %s: %s" % (i, len(arrows), arrow.name))
        if input_gen is None:
            f(arrow)
        else:
            inputs = input_gen(arrow)
            f(arrow, inputs)
