import numpy as np
from typing import Callable
from test_arrows import test_random_composite


def random_arrow_test(test: Callable,
                      test_name: str,
                      n: int=10,
                      rnd_state=None) -> None:
    """Test that function test(arrow:Arrow) works on randomly sampled arrows"""
    print("Testing ", test_name)
    """
    if rnd_state:
        np.random.set_state(rnd_state)
    else:
        rnd_state = np.random.get_state()
        print("State is")
    """

    results = []
    for i in range(n):
        # print(rnd_state)
        rand_arrow = test_random_composite()
        results.append(test(rand_arrow))
        return results
