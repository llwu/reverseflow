## There are a few different things
## - The initial inversion
## - The symbolic manipulation
## - The learning phase

def naive_invert(a: Arrow) -> Arrow:
    """Invert an arrow to construct a parametric inverse"""

def param_reduce(a: Arrow) -> Arrow:
    """Attempt to reduce the number of parameters by symbolic substitution"""


class Strategy:
    """
    A strategy describes a process transform an arrow into another
    """


class LinearStrategy(Strategy):
    """
    Applies a sequence of arrow transformations
    """

    def __init__(self, arrow_transforms: List[Callable[Arrow, Arrow]]):
        self.arrow_transforms = arrow_transforms

    def apply(a: Arrow) -> Arrow:
        for f in arrow_transforms:
            a = f(a)

        return a
