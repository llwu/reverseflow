from typing import TypeVar, Generic

L = TypeVar('L')
R = TypeVar('R')


class Bimap(Generic[L, R]):
    """Bidirectional map"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def add(self, left, right):
        self.left_to_right[left] = right
        self.right_to_left[right] = left
