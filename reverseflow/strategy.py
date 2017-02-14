from typing import List, Callable
from arrows.arrow import Arrow


# class Strategy:
#     """
#     A strategy describes a process transform an arrow into another
#     """
#
#
# class LinearStrategy(Strategy):
#     """
#     Applies a sequence of arrow transformations
#     """
#
#     def __init__(self, arrow_transforms: List[Callable[Arrow, Arrow]]) -> None:
#         self.arrow_transforms = arrow_transforms
#
#     def apply(a: Arrow) -> Arrow:
#         for f in arrow_transforms:
#             a = f(a)
#         return a
