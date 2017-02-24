"""Mapping functions.
Mapping functions are distinguished on (1) whether the relatiohip is:
OneToMany, ManyToOne. ManyToMany, OneToOne, (2) Which directions we can compute
in.


BiBij
BiRel
BiDict

For example a normal Dict is a ManyToOne mapping which can only be computed
in the forward direction.
"""
from collections import defaultdict
from typing import (TypeVar, Generic, Tuple, Set, Dict, ItemsView,
                    ValuesView, KeysView)

# TODO:
# Bidirectional or not

L = TypeVar('L')
R = TypeVar('R')


class Bimap(Generic[L, R]):
    """Bidirectional map for bijective function"""

    def __init__(self):
        self.left_to_right = {}  # type: Dict[L, R]
        self.right_to_left = {}  # type: Dict[R, L]

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def inv(self, right: R) -> L:
        return self.right_to_left[right]

    def add(self, left: L, right: R) -> None:
        self.left_to_right[left] = right
        self.right_to_left[right] = left

    def remove(self, left: L, right: R) -> None:
        if left in self.left_to_right:
            del self.left_to_right[left]
        if right in self.right_to_left:
            del self.right_to_left[right]

    def update(self, new_map: 'Bimap[L, R]') -> None:
        items = list(new_map.items())
        for (l, r) in items:
            self.add(l, r)

    def items(self):
        return self.left_to_right.items()

    def keys(self):
        return self.left_to_right.keys()

    def values(self) -> ValuesView[R]:
        return self.left_to_right.values()

    def __getitem__(self, key: L) -> R:
        return self.fwd(key)

    def __setitem__(self, key: L, value: R):
        return self.add(key, value)

    def __contains__(self, key: L):
        return key in self.left_to_right

    def __str__(self):
        return self.left_to_right.__str__()

    def __repr__(self):
        return self.left_to_right.__repr__()


class Relation(Generic[L, R]):
    """Many to Many relation"""

    def __init__(self):
        self.left_to_right = dict()  # type: Dict[L, List[R]]
        self.right_to_left = dict()  # type: Dict[R, List[L]]

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def inv(self, right: R) -> L:
        return self.right_to_left[right]

    def add(self, left: L, right: R) -> None:
        if left in self.left_to_right.keys():
            self.left_to_right[left].append(right)
        else:
            self.left_to_right[left] = [right]

        if right in self.right_to_left.keys():
            self.right_to_left[right].append(left)
        else:
            self.right_to_left[right] = [left]

    def remove(self, left: L, right: R) -> None:
        self.left_to_right[left].remove(right)
        self.right_to_left[right].remove(left)

    def items(self):
        def items_gen():
            for key, value in self.left_to_right.items():
                for v_val in value:
                    yield (key, v_val)
        return items_gen()

    def keys(self):
        return self.left_to_right.keys()

    def values(self) -> ValuesView[R]:
        return self.left_to_right.keys()

    def __getitem__(self, key: L) -> R:
        return self.fwd(key)

    def __setitem__(self, key: L, value: R):
        return self.add(key, value)

    def __contains__(self, key: L) -> bool:
        return key in self.left_to_right

    def __str__(self) -> str:
        return str(list(self.items()))

    def __repr__(self) -> str:
        return str(self)


class OneToMany(Generic[L, R]):
    """One to many relations
    Returns a set of values"""

    def __init__(self) -> None:
        self.left_to_right = {}  # type: Dict[L, Set[R]]

    def add(self, left: L, right: R) -> None:
        if left not in self.left_to_right:
            self.left_to_right[left] = set([right])
        else:
            self.left_to_right[left].add(right)

    def update(self, rel: 'OneToMany[L, R]') -> None:
        for key, val in rel.left_to_right:
            if key in self.left_to_right:
                self.left_to_right.update(val)
            else:
                self.left_to_right[key] = val

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def __getitem__(self, key: L) -> R:
        return self.fwd(key)

    def __setitem__(self, key: L, value: R) -> None:
        self.add(key, value)


class OneToManyList(Generic[L, R]):
    """One to many relations
    Returns a set of values"""

    def __init__(self) -> None:
        self.left_to_right = {}  # type: Dict[L, Set[R]]

    def add(self, left: L, right: R) -> None:
        if left not in self.left_to_right:
            self.left_to_right[left] = list([right])
        else:
            self.left_to_right[left].append(right)

    def update(self, rel: 'OneToMany[L, R]') -> None:
        for key, val in rel.left_to_right:
            if key in self.left_to_right:
                self.left_to_right.update(val)
            else:
                self.left_to_right[key] = val

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def __getitem__(self, key: L) -> R:
        return self.fwd(key)

    def __setitem__(self, key: L, value: R) -> None:
        self.add(key, value)



class ImageBimap(Generic[L, R]):
    """Bidirectional map for non-injective (many-to-one) function"""

    def __init__(self) -> None:
        self.left_to_right = {}  # type: Dict[L, R]
        self.right_to_left = {}  # type: Dict[R, Set[L]]

    def add(self, left: L, right: R) -> None:
        self.left_to_right[left] = right
        if right not in self.right_to_left:
            self.right_to_left[right] = set()
        self.right_to_left[right].add(left)

    def remove(self, left: L, right: R) -> None:
        if left in self.left_to_right:
            del self.left_to_right[left]
        if right in self.right_to_left:
            if left in self.right_to_left[right]:
                self.right_to_left[right].remove(left)

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def inv(self, right: R) -> Set[L]:
        return set(self.right_to_left[right])
