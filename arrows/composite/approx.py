from arrows.primitive.math_arrows import SubArrow, MaxArrow, ClipArrow, PowArrow
from arrows.composite.math_arrows import MeanArrow, VarFromMeanArrow, TriangleWaveArrow, ScalarVarFromMeanArrow
from arrows.compositearrow import CompositeArrow
from arrows.primitive.control_flow import DuplArrow, IfArrow, GreaterArrow, BroadcastArrow
from reverseflow.util.mapping import Bimap
from arrows.sourcearrow import SourceArrow
from arrows.port_attributes import make_error_port, make_in_port, make_out_port


class ApproxIdentityArrow(CompositeArrow):
    """Approximate Identity Arrow
    f(x_1,..,x_n) = mean(x_1,,,,.x_n), var(x_1, ..., x_n)

    Last out_port is the error port
    """

    def __init__(self, n_inputs: int, variance=ScalarVarFromMeanArrow):
        name = "ApproxIdentity"
        edges = Bimap()  # type: EdgeMap
        mean = MeanArrow(n_inputs)
        varfrommean = variance(n_inputs)
        dupls = [DuplArrow() for i in range(n_inputs)]
        for i in range(n_inputs):
            edges.add(dupls[i].out_ports()[0], mean.in_ports()[i])
            edges.add(dupls[i].out_ports()[1], varfrommean.in_ports()[i+1])
        mean_dupl = DuplArrow(n_duplications=n_inputs+1)
        edges.add(mean.out_ports()[0], mean_dupl.in_ports()[0])
        edges.add(mean_dupl.out_ports()[n_inputs], varfrommean.in_ports()[0])
        out_ports = mean_dupl.out_ports()[0:n_inputs]
        error_ports = [varfrommean.out_ports()[0]]
        # x = varfrommean.out_ports()[0]
        # error_ports = [ErrorPort(x.arrow, x.index)]
        out_ports = out_ports + error_ports
        super().__init__(edges=edges,
                         in_ports=[dupl.in_ports()[0] for dupl in dupls],
                         out_ports=out_ports,
                         name=name)
        make_error_port(self.out_ports()[-1])


class IntervalBound(CompositeArrow):

    def __init__(self, l, u):
        """
        Restricts f(x; l,u) = max(x-u, l-x, 0)
        """
        super().__init__(name="IntervalBound")
        comp_arrow = self
        in_port = comp_arrow.add_port()
        make_in_port(in_port)
        out_port = comp_arrow.add_port()
        make_out_port(out_port)
        l_src_nb = SourceArrow(l)
        u_src_nb = SourceArrow(u)
        l_src = BroadcastArrow()
        u_src = BroadcastArrow()
        comp_arrow.add_edge(l_src_nb.out_port(0), l_src.in_port(0))
        comp_arrow.add_edge(u_src_nb.out_port(0), u_src.in_port(0))

        x_min_u = SubArrow()
        comp_arrow.add_edge(in_port, x_min_u.in_ports()[0])
        comp_arrow.add_edge(u_src.out_ports()[0], x_min_u.in_ports()[1])

        l_min_x = SubArrow()
        comp_arrow.add_edge(l_src.out_ports()[0], l_min_x.in_ports()[0])
        comp_arrow.add_edge(in_port, l_min_x.in_ports()[1])

        zero = SourceArrow(0.0)
        max1 = MaxArrow()
        comp_arrow.add_edge(l_min_x.out_ports()[0], max1.in_ports()[0])
        comp_arrow.add_edge(x_min_u.out_ports()[0], max1.in_ports()[1])

        max2 = MaxArrow()
        comp_arrow.add_edge(zero.out_ports()[0], max2.in_ports()[0])
        comp_arrow.add_edge(max1.out_ports()[0], max2.in_ports()[1])
        comp_arrow.add_edge(max2.out_ports()[0], out_port)
        assert comp_arrow.is_wired_correctly()

class SmoothIntervalBound(CompositeArrow):

    def __init__(self, l, u):
        """
        Smooth version of the class IntervalBound.

        f(x; l, u) = (l - x)^2,  if x < l
        f(x; l, u) = 0,          if l <= x <= u
        f(x; l, u) = (x - u)^2,  if x > u
        """
        super().__init__(name="SmoothIntervalBound")
        comp_arrow = self
        in_port = comp_arrow.add_port()
        make_in_port(in_port)
        out_port = comp_arrow.add_port()
        make_out_port(out_port)
        l_src = SourceArrow(l)
        u_src = SourceArrow(u)

        l_gt_x = GreaterArrow()
        comp_arrow.add_edge(l_src.out_ports()[0], l_gt_x.in_ports()[0])
        comp_arrow.add_edge(in_port, l_gt_x.in_ports()[1])
        x_gt_u = GreaterArrow()
        comp_arrow.add_edge(in_port, x_gt_u.in_ports()[0])
        comp_arrow.add_edge(u_src.out_ports()[0], x_gt_u.in_ports()[1])

        two = SourceArrow(2.0)
        x_min_u = SubArrow()
        comp_arrow.add_edge(in_port, x_min_u.in_ports()[0])
        comp_arrow.add_edge(u_src.out_ports()[0], x_min_u.in_ports()[1])
        x_min_u_sqr = PowArrow()
        comp_arrow.add_edge(x_min_u.out_ports()[0], x_min_u_sqr.in_ports()[0])
        comp_arrow.add_edge(two.out_ports()[0], x_min_u_sqr.in_ports()[1])

        l_min_x = SubArrow()
        comp_arrow.add_edge(l_src.out_ports()[0], l_min_x.in_ports()[0])
        comp_arrow.add_edge(in_port, l_min_x.in_ports()[1])
        l_min_x_sqr = PowArrow()
        comp_arrow.add_edge(l_min_x.out_ports()[0], l_min_x_sqr.in_ports()[0])
        comp_arrow.add_edge(two.out_ports()[0], l_min_x_sqr.in_ports()[1])

        zero = SourceArrow(0.0)
        if1 = IfArrow()
        comp_arrow.add_edge(l_gt_x.out_ports()[0], if1.in_ports()[0])
        comp_arrow.add_edge(l_min_x_sqr.out_ports()[0], if1.in_ports()[1])
        comp_arrow.add_edge(zero.out_ports()[0], if1.in_ports()[2])

        if2 = IfArrow()
        comp_arrow.add_edge(x_gt_u.out_ports()[0], if2.in_ports()[0])
        comp_arrow.add_edge(x_min_u_sqr.out_ports()[0], if2.in_ports()[1])
        comp_arrow.add_edge(if1.out_ports()[0], if2.in_ports()[2])

        comp_arrow.add_edge(if2.out_ports()[0], out_port)
        assert comp_arrow.is_wired_correctly()

class IntervalBoundIdentity(CompositeArrow):
    """
    Identity on input but returns error for those outside bounds

    """

    def __init__(self, l, u, intervalbound=IntervalBound, clipper=ClipArrow):
        super().__init__(name="IntervalBoundIdentity")
        comp_arrow = self
        in_port = comp_arrow.add_port()
        make_in_port(in_port)
        out_port = comp_arrow.add_port()
        make_out_port(out_port)
        error_port = comp_arrow.add_port()
        make_out_port(error_port)
        make_error_port(error_port)

        l_src_nb = SourceArrow(l)
        u_src_nb = SourceArrow(u)
        l_src = BroadcastArrow()
        u_src = BroadcastArrow()
        comp_arrow.add_edge(l_src_nb.out_port(0), l_src.in_port(0))
        comp_arrow.add_edge(u_src_nb.out_port(0), u_src.in_port(0))
        interval_bound = intervalbound(l, u)
        clip = clipper()

        comp_arrow.add_edge(in_port, clip.in_ports()[0])
        comp_arrow.add_edge(l_src.out_ports()[0], clip.in_ports()[1])
        comp_arrow.add_edge(u_src.out_ports()[0], clip.in_ports()[2])
        comp_arrow.add_edge(clip.out_ports()[0], out_port)

        comp_arrow.add_edge(in_port, interval_bound.in_ports()[0])
        comp_arrow.add_edge(interval_bound.out_ports()[0], error_port)
        assert comp_arrow.is_wired_correctly()
