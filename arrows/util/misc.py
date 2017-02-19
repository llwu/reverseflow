"""Miscellaneious Utils"""
from typing import Dict, Sequence

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
