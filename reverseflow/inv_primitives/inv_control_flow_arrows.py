from reverseflow.arrows.primitivearrow import PrimitiveArrow


class InvDuplArrow(PrimitiveArrow):
    """InvDupl f(x,x) = x"""

    def __init__(self):
        name = "InvDupl"
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)
