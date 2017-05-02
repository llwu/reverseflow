"""
This file contains random variable arrows with closed form quantile functions
for sampling. For any distribution with a quantile function Q, one can sample
from the distribution by taking Q(p), where p ~ U[0, 1].
"""
from arrows.compositearrow import CompositeArrow
from arrows.primitive.math_arrows import *
from arrows.sourcearrow import SourceArrow
from arrows.port_attributes import make_in_port, make_out_port, make_param_port
from arrows.apply.apply import apply
from reverseflow.invert import invert
from reverseflow.inv_primitives.inv_math_arrows import *
import numpy as np

class ExponentialRV():
    def __init__(self, lmbda: float) -> None:
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
        comp_arrow.add_edge(lmbda_source.out_ports()[0], div_lmbda.in_ports()[1])
        comp_arrow.add_edge(div_lmbda.out_ports()[0], out_port)

        assert comp_arrow.is_wired_correctly()
        self.quantile = comp_arrow

    def sample(self, shape = [1]):
        """
        Sample from the given exponential distribution with parameter lmbda
        using the closed form quantile function. The samples will be returned
        in the given shape (must be iterable).
        """

        size = 1
        for shape_dim in shape:
            size = size * shape_dim
        p_values = np.random.uniform(size = size)
        samples = []
        for p in p_values:
            sample = apply(self.quantile, [p])
            samples.append(sample[0])
        samples = np.asarray(samples)
        return np.reshape(samples, shape)


"""
===============================================================================
=====================  Experimental code ======================================
===============================================================================
"""
"""
Consider the forward function f(x_1, x_2) = abs(x_1 - x_2).
Its parametric inverse is f^(-1)(y; theta_1, theta_2) = (theta_1*y+theta_2, theta_2),
where theta_1 = sign(x_1 - x_2) and theta_2 = x_2.
For generating data we take x_1, x_2 ~ Exp(lmbda).
"""
def gen_data(lmbda: float, size: int):
    forward_arrow = CompositeArrow(name="forward")
    in_port1 = forward_arrow.add_port()
    make_in_port(in_port1)
    in_port2 = forward_arrow.add_port()
    make_in_port(in_port2)
    out_port = forward_arrow.add_port()
    make_out_port(out_port)

    subtraction = SubArrow()
    absolute = AbsArrow()
    forward_arrow.add_edge(in_port1, subtraction.in_ports()[0])
    forward_arrow.add_edge(in_port2, subtraction.in_ports()[1])
    forward_arrow.add_edge(subtraction.out_ports()[0], absolute.in_ports()[0])
    forward_arrow.add_edge(absolute.out_ports()[0], out_port)
    assert forward_arrow.is_wired_correctly()

    inverse_arrow = CompositeArrow(name="inverse")
    in_port = inverse_arrow.add_port()
    make_in_port(in_port)
    param_port_1 = inverse_arrow.add_port()
    make_in_port(param_port_1)
    make_param_port(param_port_1)
    param_port_2 = inverse_arrow.add_port()
    make_in_port(param_port_2)
    make_param_port(param_port_2)
    out_port_1 = inverse_arrow.add_port()
    make_out_port(out_port_1)
    out_port_2 = inverse_arrow.add_port()
    make_out_port(out_port_2)

    inv_sub = InvSubArrow()
    inv_abs = InvAbsArrow()
    inverse_arrow.add_edge(in_port, inv_abs.in_ports()[0])
    inverse_arrow.add_edge(param_port_1, inv_abs.in_ports()[1])
    inverse_arrow.add_edge(inv_abs.out_ports()[0], inv_sub.in_ports()[0])
    inverse_arrow.add_edge(param_port_2, inv_sub.in_ports()[1])
    inverse_arrow.add_edge(inv_sub.out_ports()[0], out_port_1)
    inverse_arrow.add_edge(inv_sub.out_ports()[1], out_port_2)
    assert inverse_arrow.is_wired_correctly()

    eps = 1e-5
    data = []
    dist = ExponentialRV(lmbda)
    for _ in range(size):
        x = tuple(dist.sample(shape=[2]))
        y = np.abs(x[0]-x[1])
        theta = (np.sign(x[0] - x[1]), x[1])
        y_ = apply(forward_arrow, list(x))
        x_ = apply(inverse_arrow, [y, theta[0], theta[1]])
        assert np.abs(y_[0] - y) < eps
        assert np.abs(x_[0] - x[0]) < eps
        assert np.abs(x_[1] - x[1]) < eps
        data.append((x, theta, y))
    return data, forward_arrow, inverse_arrow
