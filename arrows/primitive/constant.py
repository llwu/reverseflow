from arrows.primitivearrow import PrimitiveArrow


class RangeArrow(PrimitiveArrow):
    """range"""

    def __init__(self):
        name = 'range [a,b]'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)


class RankArrow(PrimitiveArrow):
    """Number of dimensions of an arrow"""

    def __init__(self):
        name = 'rank'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)
