"""Miscellaneious Utils"""
import itertools
import string
import random
from typing import Dict, Sequence, Any


def rand_string(n):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def extract(keys: Sequence, dict: Dict):
    """Restrict dict to keys in `keys`"""
    return {key: dict[key] for key in keys}


def getn(dict: Dict, *keys):
    return (dict[k] for k in keys)


def inn(seq, *keys):
    return all((k in seq for k in keys))


def print_one_per_line(xs:Sequence):
    """Simple printing of one element of xs per line"""
    for x in xs:
        print(x)


def same(xs) -> bool:
    """All elements in xs are the same?"""
    if len(xs) == 0:
        return True
    else:
        x1 = xs[0]
        for xn in xs:
            if xn != x1:
                return False

    return True


def dict_prod(d: Dict[Any, Sequence]):
    """
    Cartesian product for dictionaries
    Args:
        dict: A dictionary where all keys are iterable
    Returns:
        iterable over dict
    """
    keys = d.keys()
    it = itertools.product(*d.values())
    return (dict(zip(keys, tup)) for tup in it)
