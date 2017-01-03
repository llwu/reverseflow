"""Functions to generate the arrows for tests."""

from random import randint, sample
import inspect

import tensorflow as tf

import reverseflow.arrows.primitive.math_arrows
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.inv_primitives.inv_math_arrows import *
from reverseflow.inv_primitives.inv_control_flow_arrows import *
from reverseflow.arrows.composite.approx import *
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.arrows.primitive.cast_arrows import *
from reverseflow.arrows.primitivearrow import PrimitiveArrow
from reverseflow.arrows.sourcearrow import SourceArrow
from reverseflow.arrows.compose import compose_comb_modular, compose_comb
from reverseflow.defaults import default_dispatch
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.config import floatX


def test_xyplusx_flat() -> CompositeArrow:
    """f(x,y) = x * y + x"""
    mul = MulArrow()
    add = AddArrow()
    dupl = DuplArrow()
    edges = Bimap()  # type: EdgeMap
    edges.add(dupl.out_ports[0], mul.in_ports[0])  # dupl -> mul
    edges.add(dupl.out_ports[1], add.in_ports[0])  # dupl -> add
    edges.add(mul.out_ports[0], add.in_ports[1])  # mul -> add
    d = CompositeArrow(in_ports=[dupl.in_ports[0], mul.in_ports[1]],
                       out_ports=[add.out_ports[0]], edges=edges)
    return d


def test_xyplusx() -> CompositeArrow:
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

def test_inv_twoxyplusx() -> CompositeArrow:
    """approximate parametric inverse of twoxyplusx"""
    inv_add = InvAddArrow()
    inv_mul = InvMulArrow()
    two_int = SourceArrow(2)
    two = CastArrow(floatX())
    div = DivArrow()
    c = ApproxIdentityArrow(2)
    inv_dupl = InvDuplArrow()
    edges = Bimap()  # type: EdgeMap
    edges.add(two_int.out_ports[0], two.in_ports[0])
    edges.add(inv_add.out_ports[0], c.in_ports[0])
    edges.add(inv_add.out_ports[1], inv_mul.in_ports[0])
    edges.add(inv_mul.out_ports[0], div.in_ports[0])
    edges.add(two.out_ports[0], div.in_ports[1])
    edges.add(div.out_ports[0], c.in_ports[1])
    edges.add(c.out_ports[0], inv_dupl.in_ports[0])
    edges.add(c.out_ports[1], inv_dupl.in_ports[1])

    param_inports = [inv_add.in_ports[1], inv_mul.in_ports[1]]
    op = CompositeArrow(in_ports=[inv_add.in_ports[0]] + param_inports,
                       out_ports=[inv_dupl.out_ports[0], inv_mul.out_ports[1], c.out_ports[2]],
                       edges=edges)
    op.change_in_port_type(ParamPort, 1)
    op.change_in_port_type(ParamPort, 2)
    op.change_out_port_type(ErrorPort, 2)
    op.name = "InvTwoXPlusY"
    return op



def test_multicomb() -> CompositeArrow:
    """f(a,b,c,d,e,f) = (a+b)(cd(e+f) + e + f)"""
    add1, add2 = AddArrow(), AddArrow()
    mul1, mul2 = MulArrow(), MulArrow()
    comp = test_xyplusx()
    mul_comp = compose_comb_modular(mul1, comp, {0: 0})
    mul_add_comp = compose_comb(add1, mul_comp, {0: 2})
    add_mul = compose_comb(add2, mul2, {0: 0})
    multicomb = compose_comb(mul_add_comp, add_mul, {0: 2})
    return multicomb


def test_random_math() -> PrimitiveArrow:
    """Generates a random math arrow."""

    # FIXME: Hack until we have more general random generator
    # maths = [AddArrow,
    #          SubArrow,
    #          MulArrow,
    #          DivArrow,
    #          PowArrow,
    #          ExpArrow,
    #          LogArrow,
    #          LogBaseArrow,
    #          NegArrow,
    #          AddNArrow,
    #          AbsArrow]

    maths = [m[1] for m in inspect.getmembers(reverseflow.arrows.primitive.math_arrows,
            inspect.isclass) if m[1].__module__ == 'reverseflow.arrows.primitive.math_arrows']
    maths = list(set.intersection(set(maths), default_dispatch.keys()))
    idx = randint(0, len(maths) - 1)
    args = []
    n_input_arrows = [
        reverseflow.arrows.primitive.math_arrows.ReduceMeanArrow,
        reverseflow.arrows.primitive.math_arrows.AddNArrow
        ]
    if maths[idx] in n_input_arrows:
        args = [randint(1, 5)]
    return maths[idx](*args)


def test_random_input() -> Arrow:
    """Generates a random math or source arrow."""
    odds = 50000000
    if randint(0, odds) == 0:
        return SourceArrow(randint(0, 0xBADA55))
    else:
        return test_random_math()


def test_random_composite() -> CompositeArrow:
    """Generates a random arrow."""
    min_size = 10
    max_size = 50
    arrow = test_random_math()
    size = randint(min_size, max_size)
    for _ in range(size):
        if arrow.num_in_ports() > 1:
            odds = 5
            if randint(0, odds) == 0:
                ports = sample(range(arrow.num_in_ports()), 2)
                arrow = compose_comb(
                    DuplArrow(),
                    arrow,
                    {0: ports[0], 1: ports[1]})
            else:
                arrow = compose_comb(
                    test_random_input(),
                    arrow,
                    {0: randint(0, arrow.num_in_ports() - 1)})
        else:
            arrow = compose_comb(
                test_random_math(),
                arrow,
                {0: randint(0, arrow.num_in_ports() - 1)})
    return arrow

# FIXME: There must be a better way to get all arrows
import reverseflow.inv_primitives.inv_math_arrows
import reverseflow.arrows.composite.math
composite_module_list = [reverseflow.inv_primitives.inv_math_arrows]

def all_composites() -> List:
    composites = []
    for module in composite_module_list:
        comp_arrow_classes = [m[1] for m in inspect.getmembers(module,
                  inspect.isclass) if inspect.getmodule(m[1]) == module]
        composites = composites + comp_arrow_classes
    return composites
