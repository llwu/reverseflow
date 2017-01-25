from arrows import Arrow
from typing import Sequence, Callable

def totality_test(f:Callable,
                  arrows: Sequence[Arrow],
                  input_gen: Callable=None,
                  ignore: Callable=None,
                  test_name: str=None,
                  **kwargs):
    """Helper to check that arrow simply runs.
    args:
        f: a function which takes an arrow as its first input
        arrow: An arrow to test
        input_gen: function which generates input from arrow, if None
                   then no input is supplied to testing f
        ignore: function which takes arrow and returns true if we should ignore
    """
    print("Testing %s" % test_name)
    for i, arrow in enumerate(arrows):
        if ignore is not None:
            if ignore(arrow):
                print("Skipping arrow:", arrow.name)
                continue
        print("Arrow %s of %s: %s" % (i+1, len(arrows), arrow.name))
        if input_gen is None:
            f(arrow)
        else:
            inputs = input_gen(arrow)
            f(arrow, inputs, **kwargs)
