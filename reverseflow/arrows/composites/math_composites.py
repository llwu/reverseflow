from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.primitive.math_arrows import AddNArrow


class MeanArrow(CompositeArrow):
    """
    Takes in n tensors of same shape and returns one tensor of elementwise mean
    """

    def __init__(self, n_inputs: int):
        name = "Mean"
        addn_arrow = AddNArrow(n_inputs)
        nsource = SourceArrow(n_inputs)
        div = DivArrow()
        div_n = compose_comb(nsource, div_arrow, {0: 1})
        return compose_comb(addn_arrow, div_n, {0: 0}, name=name)
