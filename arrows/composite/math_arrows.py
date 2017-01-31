from arrows.compositearrow import CompositeArrow
from arrows.primitive.control_flow import *
from arrows.primitive.math_arrows import *
from arrows.primitive.constant import *
from arrows.primitive.cast_arrows import *
from arrows.sourcearrow import SourceArrow
from reverseflow.util.mapping import Bimap
from arrows.compose import compose_comb
from arrows.config import floatX


class DimsBarBatchArrow(CompositeArrow):
    def __init__(self) -> None:
        name = 'DimsBarBatch'
        rank_arrow = RankArrow()
        one_source = SourceArrow(1)
        range_arrow = RangeArrow()
        edges = Bimap()  #  type: EdgeMap
        edges.add(one_source.get_out_ports()[0], range_arrow.get_in_ports()[0])
        edges.add(rank_arrow.get_out_ports()[0], range_arrow.get_in_ports()[1])
        super().__init__(edges=edges,
                         in_ports=rank_arrow.get_in_ports(),
                         out_ports=range_arrow.get_out_ports(),
                         name=name)


class MeanArrow(CompositeArrow):
    """
    Takes in n tensors of same shape and returns one tensor of elementwise mean
    """

    def __init__(self, n_inputs: int) -> None:
        name = 'Mean'
        edges = Bimap() # type: EdgeMap
        addn_arrow = AddNArrow(n_inputs)
        nsource = SourceArrow(n_inputs)
        castarrow = CastArrow(floatX())
        div_arrow = DivArrow()
        edges.add(nsource.get_out_ports()[0], castarrow.get_in_ports()[0])
        edges.add(addn_arrow.get_out_ports()[0], div_arrow.get_in_ports()[0])
        edges.add(castarrow.get_out_ports()[0], div_arrow.get_in_ports()[1])
        super().__init__(edges=edges,
                         in_ports=addn_arrow.get_in_ports(),
                         out_ports=div_arrow.get_out_ports(),
                         name=name)


class VarFromMeanArrow(CompositeArrow):
    """
    Compute variance given mean and set of inputs
    In_port 0: mean
    in_port 1 .. n+1: values to compute variance
    """

    def __init__(self, n_inputs: int) -> None:
        name = 'VarFromMean'
        # import pdb; pdb.set_trace()
        dupl = DuplArrow(n_duplications=n_inputs)
        subs = [SubArrow() for i in range(n_inputs)]
        abss = [AbsArrow() for i in range(n_inputs)]
        addn = AddNArrow(n_inputs)

        edges = Bimap()  # type: EdgeMap
        in_ports = [dupl.get_in_ports()[0]] + [sub.get_in_ports()[1] for sub in subs]
        for i in range(n_inputs):
            edges.add(dupl.get_out_ports()[i], subs[i].get_in_ports()[0])
            edges.add(subs[i].get_out_ports()[0], abss[i].get_in_ports()[0])
            edges.add(abss[i].get_out_ports()[0], addn.get_in_ports()[i])

        dupl2 = DuplArrow(n_duplications=2)
        edges.add(addn.get_out_ports()[0], dupl2.get_in_ports()[0])

        reduce_mean = ReduceMeanArrow(n_inputs=2)
        dimsbarbatch = DimsBarBatchArrow()

        edges.add(dupl2.get_out_ports()[0], reduce_mean.get_in_ports()[0])
        edges.add(dupl2.get_out_ports()[1], dimsbarbatch.get_in_ports()[0])
        edges.add(dimsbarbatch.get_out_ports()[0], reduce_mean.get_in_ports()[1])
        out_ports = reduce_mean.get_out_ports()

        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         name=name)
