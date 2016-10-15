<<<<<<< HEAD
from typing import TypeVar, Generic, Set
=======
from typing import TypeVar, Generic, Tuple, List, Set
>>>>>>> 883f5c95d9e2f58176cc72ad88b57167af8fb318

L = TypeVar('L')
R = TypeVar('R')


class Bimap(Generic[L, R]):
    """Bidirectional and bijective map"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def fwd(self, left: L) -> R:
        if left not in self.left_to_right:
            return None
        return self.left_to_right[left]

    def inv(self, right: R) -> L:
        if right not in self.right_to_left:
            return None
        return self.right_to_left[right]

    def add(self, left: L, right: R) -> None:
        self.left_to_right[left] = right
        self.right_to_left[right] = left

    def remove(self, left: L, right: R) -> None:
        if left in self.left_to_right:
            del self.left_to_right[left]
        if right in self.right_to_left:
            del self.right_to_left[right]

    def items(self) -> List[Tuple[L, R]]:
        return self.left_to_right.items()


class ImageBimap(Generic[L, R]):
    """Bidirectional map for non-injective function"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

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
