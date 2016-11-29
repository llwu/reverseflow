from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.composites.math_composites import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.constant import *
from reverseflow.arrows.sourcearrow import SourceArrow
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compose import compose_comb

# import numpy as np
#
# def dims_bar_batch(t):
#     """Get dimensions of a tensor exluding its batch dimension (first one)"""
#     return np.arange(1, t.get_shape().ndims)


class DimsBarBatchArrow():
    def __init__(self):
        name = "DimsBarBatch"
        rank_arrow = RankArrow()
        one_source = SourceArrow(1)
        range_arrow = RangeArrow()
        edges = Bimap()  #  type: EdgeMap
        edges.add(one_source.out_ports[0], range_arrow.in_ports[0])
        edges.add(rank_arrow.out_ports[1], range_arrow.in_ports[1])
        return super().__init__(edges=edges,
                                in_ports=rank_arrow.in_ports,
                                out_ports=range_arrow.out_ports,
                                name=name)


class MeanArrow(CompositeArrow):
    """
    Takes in n tensors of same shape and returns one tensor of elementwise mean
    """

    def __init__(self, n_inputs: int) -> None:
        name = "Mean"
        addn_arrow = AddNArrow(n_inputs)
        nsource = SourceArrow(n_inputs)
        div_arrow = DivArrow()
        div_n = compose_comb(nsource, div_arrow, {0: 1})
        mean_arrow = compose_comb(addn_arrow, div_n, {0: 0}, name=name)
        super().__init__(edges=mean_arrow.edges,
                         in_ports=mean_arrow.inner_in_ports(),
                         out_ports=mean_arrow.inner_out_ports(),
                         name=name)


class VarFromMean(CompositeArrow):
    """
    Compute variance given variance and set of inputs
    """

def __init__(self, n_inputs: int) -> None:
    name = "VarFromMean"
    dupl = DuplArrow(n_duplications=n_inputs)
    subs = [SubArrow() for i in range(n_inputs)]
    abss = [AbsArrow() for i in range(n_inputs)]
    addn = AddNArrow(n_inputs)

    edges = Bimap()  # type: EdgeMap
    in_ports = [dupl.in_ports[0]] + [sub.in_ports[1] for sub in subs]
    for i in range(n_inputs):
        edges.add(dupl.out_ports[i], subs[i].in_ports[i])
        edges.add(subs[i].out_ports[0], abss[i].in_ports[0])
        edges.add(abss[i].out_ports[0], addn.in_ports[i])

    dupl2 = DuplArrow(n_duplications=2)
    edges.add(addn.out_ports[0], dupl2.in_ports[0])

    reduce_mean = ReduceMeanArrow(reduction_indices=dims_bar_batch(mean_variances))
    dimsbarbatch = DimsBarBatchArrow()

    edges.add(dupl2.out_ports[0], reduce_mean.in_ports[0])
    edges.add(dupl2.out_ports[1], dimsbarbatch.in_ports[0])
    edges.add(dimsbarbatch.out_ports[0], reduce_mean.in_ports[1])
    out_ports = reduce_mean.out_ports
    super().__init__(edges=edges,
                     in_ports=in_ports,
                     out_ports=out_ports,
                     name=name)
