"""Mapping functions"""
from typing import (TypeVar, Generic, Tuple, Set, Dict, ItemsView,
                    ValuesView, KeysView)

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

    def items(self) -> ItemsView[L, R]:
        return self.left_to_right.items()

    def keys(self) -> KeysView[L]:
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
        return self.left_to_right.__str__()


class ImageBimap(Generic[L, R]):
    """Bidirectional map for non-injective function"""

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
