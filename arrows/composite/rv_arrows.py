"""
This file contains random variable arrows with closed form quantile functions
for sampling. For any distribution with a quantile function Q, one can sample
from the distribution by taking Q(p), where p ~ U[0, 1].
"""
import numpy as np

class ExponentialRV():
    def __init__(self, size: int, lmbda: float) -> None:
        self.size = size
        comp_arrow = CompositeArrow(name="ExponentialRVQuantile")
        in_port = comp_arrow.add_port()
        make_in_port(in_port)
        out_port = comp_arrow.add_port()
        make_out_port(out_port)

        lmbda_source = SourceArrow(lmbda)
        one_source = SourceArrow(1.0)
        one_minus_p = SubArrow()
        comp_arrow.add_edge(one_source.out_ports()[0], one_minus_p.in_ports()[0])
        comp_arrow.add_edge(in_port, one_minus_p.in_ports()[1])
        ln = LogArrow()
        comp_arrow.add_edge(one_minus_p.out_ports()[0], ln.in_ports()[0])

        negate = NegArrow()
        comp_arrow.add_edge(ln.out_ports()[0], negate.in_ports()[0])
        div_lmbda = DivArrow()
        comp_arrow.add_edge(negate.out_ports()[0], div_lmbda.in_ports()[0])
        comp_arrow.add_edge(lmbda_source.in_ports()[0], div_lmbda.in_ports()[1])
        comp_arrow.add_edge(div_lmbda.out_ports()[0], out_port)

        assert comp_arrow.is_wired_correctly()
        self.quantile = comp_arrow

    def sample():
        p = np.random.uniform(size = self.size)
        samples = []
        for p_value in p:
            sample = propagate(self.quantile, [p_value])  # FIXME find the actual corresponding method
            samples.append(sample)
        return samples
