from typing import TypeVar, Generic

L = TypeVar('L')
R = TypeVar('R')


class Bimap(Generic[L, R]):
    """Bidirectional and bijective map"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def add(self, left: L, right: R):
        self.left_to_right[left] = right
        self.right_to_left[right] = left

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def inv(self, right: R) -> L:
        return self.right_to_left[right]

class ImageBimap(Generic[L, R]):
    """Bidirectional map for non-injective function"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def add(self, left: L, right: R):
        self.left_to_right[left] = right
        if right not in self.right_to_left:
            self.right_to_left[right] = Set()
        self.right_to_left[right].add(left)

    def fwd(self, left: L) -> R:
        return self.left_to_right[left]

    def inv(self, right: R) -> Set[L]:
        return self.right_to_left[right]
