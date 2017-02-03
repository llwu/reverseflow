from arrows.compositearrow import CompositeArrow
from arrows.primitive.control_flow import *
from arrows.primitive.math_arrows import *
from arrows.primitive.constant import *
from arrows.primitive.cast_arrows import *
from arrows.sourcearrow import SourceArrow
from reverseflow.util.mapping import Bimap
from arrows.compose import compose_comb
from arrows.config import floatX
from arrows.port_attributes import make_in_port, make_out_port


class DimsBarBatchArrow(CompositeArrow):
    def __init__(self) -> None:
        name = 'DimsBarBatch'
        rank_arrow = RankArrow()
        one_source = SourceArrow(1)
        range_arrow = RangeArrow()
        edges = Bimap()  #  type: EdgeMap
        edges.add(one_source.out_ports()[0], range_arrow.in_ports()[0])
        edges.add(rank_arrow.out_ports()[0], range_arrow.in_ports()[1])
        super().__init__(edges=edges,
                         in_ports=rank_arrow.in_ports(),
                         out_ports=range_arrow.out_ports(),
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
        edges.add(nsource.out_ports()[0], castarrow.in_ports()[0])
        edges.add(addn_arrow.out_ports()[0], div_arrow.in_ports()[0])
        edges.add(castarrow.out_ports()[0], div_arrow.in_ports()[1])
        super().__init__(edges=edges,
                         in_ports=addn_arrow.in_ports(),
                         out_ports=div_arrow.out_ports(),
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
        in_ports = [dupl.in_ports()[0]] + [sub.in_ports()[1] for sub in subs]
        for i in range(n_inputs):
            edges.add(dupl.out_ports()[i], subs[i].in_ports()[0])
            edges.add(subs[i].out_ports()[0], abss[i].in_ports()[0])
            edges.add(abss[i].out_ports()[0], addn.in_ports()[i])

        dupl2 = DuplArrow(n_duplications=2)
        edges.add(addn.out_ports()[0], dupl2.in_ports()[0])

        reduce_mean = ReduceMeanArrow(n_inputs=2)
        dimsbarbatch = DimsBarBatchArrow()

        edges.add(dupl2.out_ports()[0], reduce_mean.in_ports()[0])
        edges.add(dupl2.out_ports()[1], dimsbarbatch.in_ports()[0])
        edges.add(dimsbarbatch.out_ports()[0], reduce_mean.in_ports()[1])
        out_ports = reduce_mean.out_ports()

        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         name=name)

class ScalarVarFromMeanArrow(CompositeArrow):
    """
    Compute variance given mean and set of scalar inputs
    In_port 0: mean
    in_port 1 .. n+1: values to compute variance
    """

    def __init__(self, n_inputs: int) -> None:
        super().__init__(name="ScalarVarFromMean")
        comp_arrow = self
        in_ports = [comp_arrow.add_port() for i in range(n_inputs + 1)]
        for in_port in in_ports:
            make_in_port(in_port)
        out_port = comp_arrow.add_port()
        make_out_port(out_port)

        sub_arrows = [SubArrow() for i in range(n_inputs)]
        two = SourceArrow(2.0)
        squares = [PowArrow() for i in range(n_inputs)]
        addn = AddNArrow()
        for i in range(n_inputs):
            comp_arrow.add_edge(in_ports[0], sub_arrows[i].in_ports()[1])
            comp_arrow.add_edge(in_ports[i + 1], sub_arrows[i].in_ports()[0])
            comp_arrow.add_edge(sub_arrows[i].out_ports()[0], squares[i].in_ports()[0])
            comp_arrow.add_edge(two.out_ports()[0], squares[i].in_ports()[1])
            comp_arrow.add_edge(squares[i].out_ports()[0], addn.in_ports()[i])

        nn = SourceArrow(n_inputs)
        cast = CastArrow(floatX())
        variance = DivArrow()
        comp_arrow.add_edge(nn.out_ports()[0], cast.in_ports()[0])
        comp_arrow.add_edge(addn.out_ports()[0], variance.in_ports()[0])
        comp_arrow.add_edge(cast.out_ports()[0], variance.in_ports()[1])

        comp_arrow.add_edge(variance.out_ports()[0], out_port)
        assert comp_arrow.is_wired_correctly


class TriangleWaveArrow(CompositeArrow):
    """
    Compute the triangle wave function with bounds (l, u)
    as the last two inputs to the arrow.
    In_port 0: input argument - a
    In_port 1: lower bound - l
    In_port 2: upper bound - u
    Formula: t = a - 2 * (u - l) * floordiv(a, 2 * (u - l)) + l
             return t > u ? 2 * u - t : t
    """

    def __init__(self) -> None:
        super().__init__(name="TriangleWave")
        comp_arrow = self
        in_port0 = comp_arrow.add_port()
        make_in_port(in_port0)
        in_port1 = comp_arrow.add_port()
        make_in_port(in_port1)
        in_port2 = comp_arrow.add_port()
        make_in_port(in_port2)
        out_port = comp_arrow.add_port()
        make_out_port(out_port)

        two = SourceArrow(2.0)
        u_minus_l = SubArrow()
        comp_arrow.add_edge(in_port2, u_minus_l.in_ports()[0])
        comp_arrow.add_edge(in_port1, u_minus_l.in_ports()[1])
        twice_u_minus_l = MulArrow()
        comp_arrow.add_edge(u_minus_l.out_ports()[0], twice_u_minus_l.in_ports()[0])
        comp_arrow.add_edge(two.out_ports()[0], twice_u_minus_l.in_ports()[1])

        floordiv = FloorDivArrow()
        comp_arrow.add_edge(in_port0, floordiv.in_ports()[0])
        comp_arrow.add_edge(twice_u_minus_l.out_ports()[0], floordiv.in_ports()[1])
        product = MulArrow()
        comp_arrow.add_edge(floordiv.out_ports()[0], product.in_ports()[0])
        comp_arrow.add_edge(twice_u_minus_l.out_ports()[0], product.in_ports()[1])
        a_sub = SubArrow()
        comp_arrow.add_edge(in_port0, a_sub.in_ports()[0])
        comp_arrow.add_edge(product.out_ports()[0], a_sub.in_ports()[1])
        t = AddArrow()
        comp_arrow.add_edge(a_sub.out_ports()[0], t.in_ports()[0])
        comp_arrow.add_edge(in_port1, t.in_ports()[1])

        t_gt_u = GreaterArrow()
        comp_arrow.add_edge(t.out_ports()[0], t_gt_u.in_ports()[0])
        comp_arrow.add_edge(in_port2, t_gt_u.in_ports()[1])
        twice_u = MulArrow()
        comp_arrow.add_edge(in_port2, twice_u.in_ports()[0])
        comp_arrow.add_edge(two.out_ports()[0], twice_u.in_ports()[1])
        twou_minus_t = SubArrow()
        comp_arrow.add_edge(twice_u.out_ports()[0], twou_minus_t.in_ports()[0])
        comp_arrow.add_edge(t.out_ports()[0], twou_minus_t.in_ports()[1])

        if_arrow = IfArrow()
        comp_arrow.add_edge(t_gt_u.out_ports()[0], if_arrow.in_ports()[0])
        comp_arrow.add_edge(twou_minus_t.out_ports()[0], if_arrow.in_ports()[1])
        comp_arrow.add_edge(t.out_ports()[0], if_arrow.in_ports()[2])
        comp_arrow.add_edge(if_arrow.out_ports()[0], out_port)
        assert comp_arrow.is_wired_correctly()
