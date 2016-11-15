"""Functions to generate the arrows for tests."""

from reverseflow.arrows.primitive.math_arrows import MulArrow, AddArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.arrows.sourcearrow import SourceArrow
from reverseflow.arrows.compose import compose_comb_modular, compose_comb
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
import tensorflow as tf


def test_xyplusx() -> None:
    """f(x,y) = x * y + x"""
    tf.reset_default_graph()
    mul = MulArrow()
    add = AddArrow()
    dupl = DuplArrow()
    dupl_mul = compose_comb(dupl, mul, {0: 0})
    dupl_mul_add = compose_comb(dupl_mul, add, {0: 0, 1: 1})
    return dupl_mul_add


def test_twoxyplusx() -> CompositeArrow:
    """f(x,y) = 2 * x * y + x"""
    tf.reset_default_graph()
    two = SourceArrow(2)
    mul1 = MulArrow()
    mul2 = MulArrow()
    add = AddArrow()
    dupl = DuplArrow()
    edges = Bimap()  # type: EdgeMap
    edges.add(dupl.out_ports[0], mul1.in_ports[0])  # dupl -> mul1
    edges.add(dupl.out_ports[1], add.in_ports[0])  # dupl -> add
    edges.add(two.out_ports[0], mul2.in_ports[0])
    edges.add(mul1.out_ports[0], mul2.in_ports[1])
    edges.add(mul2.out_ports[0], add.in_ports[1])  # mul1 -> add
    return CompositeArrow(in_ports=[dupl.in_ports[0], mul1.in_ports[1]],
                          out_ports=[add.out_ports[0]],
                          edges=edges)


def test_multicomb() -> None:
    """f(a,b,c,d,e,f) = (a+b)(cd(e+f) + e + f)"""
    add1, add2 = AddArrow(), AddArrow()
    mul1, mul2 = MulArrow(), MulArrow()
    comp = test_xyplusx()
    mul_comp = compose_comb_modular(mul1, comp, {0: 0})
    mul_add_comp = compose_comb(add1, mul_comp, {0: 2})
    add_mul = compose_comb(add2, mul2, {0: 0})
    multicomb = compose_comb(mul_add_comp, add_mul, {0: 2})
    return multicomb
