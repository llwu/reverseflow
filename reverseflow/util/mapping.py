from typing import TypeVar, Generic

L = TypeVar('L')
R = TypeVar('R')


class Bimap(Generic[L, R]):
    """Bidirectional and bijective map"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def add(self, left: T, right: T) -> None:
        self.left_to_right[left] = right
        self.right_to_left[right] = left

    def items(self):
        return self.left_to_right.items()

class ImageBimap(Generic[L, R]):
    """Bidirectional map for non-injective function"""

    def __init__(self):
        # TODO
        pass

    def fwd(l: L) -> R:
        # TODO
        pass

    def inv(r: R) -> Set[L]:
        # TODO
        pass
