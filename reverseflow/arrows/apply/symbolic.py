"""Symbolically Evaluate a Graph"""
from collections import OrderedDict
from typing import Dict, Sequence, MutableMapping, Any
from reverseflow.arrows.port import InPort, OutPort
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.primitive.math_arrows import (AddArrow,
    MulArrow, DivArrow)
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow


from pqdict import pqdict
import sympy
from sympy import Expr
from overloading import overload

ExprList = Sequence[Expr]

@overload
def conv(add: AddArrow, args: ExprList) -> ExprList:
    assert len(args) == 2
    return [args[0] + args[1]]


@overload
def conv(a: MulArrow, args: ExprList) -> ExprList:
    assert len(args) == 2
    return [args[0] * args[1]]


@overload
def conv(dupl: DuplArrow, args: ExprList) -> ExprList:
    return [args[0] for i in range(dupl.n_out_ports)]


def default_add(arrow_exprs: Dict[Arrow, MutableMapping[int, Expr]],
                sub_arrow: Arrow, index: int, input_expr: Expr) -> None:
    if sub_arrow in arrow_exprs:
        arrow_exprs[sub_arrow][index] = input_expr
    else:
        arrow_exprs[sub_arrow] = OrderedDict({index: input_expr})


def print_arrow_colors(arrow_colors):
    for (arr, pr) in arrow_colors.items():
        print(arr.name, ": ", pr)


@overload
def symbolic_apply(comp_arrow) -> Dict[Arrow, MutableMapping[int, Expr]]:
    """Convert an comp_arrow to a tensorflow graph"""

    # A priority queue for each sub_arrow
    # priority is the number of inputs it has which have already been seen
    # seen inputs are inputs to the composition, or outputs of arrows that
    # have already been converted into tensorfow
    arrow_colors = pqdict()
    # import pdb; pdb.set_trace()
    for sub_arrow in comp_arrow.get_sub_arrows():
        arrow_colors[sub_arrow] = sub_arrow.num_in_ports()

    print_arrow_colors(arrow_colors)

    # Store a map from an arrow to its inputs
    # Use a dict because no guarantee we'll create input tensors in order
    arrow_exprs = dict()  # type: Dict[Arrow, MutableMapping[int, Expr]]

    # create a tensor for each in_port to the composition
    # decrement priority for each arrow connected to inputs
    for i, in_port in enumerate(comp_arrow.in_ports):
        sub_arrow = in_port.arrow
        assert sub_arrow in arrow_colors
        arrow_colors[sub_arrow] = arrow_colors[sub_arrow] - 1
        input_expr = sympy.var('inp_%s' % i)
        default_add(arrow_exprs, sub_arrow, in_port.index, input_expr)

    while len(arrow_colors) > 0:
        print_arrow_colors(arrow_colors)
        sub_arrow, priority = arrow_colors.popitem()
        print("Converting ", sub_arrow.name)
        print_arrow_colors(arrow_colors)
        assert priority == 0, "Must resolve all inputs to sub_arrow first"
        assert sub_arrow.is_primitive(), "Cannot convert unflat arrow"
        # assert valid(sub_arrow, arrow_exprs)

        inputs = list(arrow_exprs[sub_arrow].values())
        # import pdb; pdb.set_trace()
        outputs = conv(sub_arrow, inputs)
        assert len(outputs) == len(sub_arrow.out_ports), "diff num outputs"

        for i, out_port in enumerate(sub_arrow.out_ports):
            # FIXME: this is linear search, encapsulate
            if out_port not in comp_arrow.out_ports:
                neigh_port = comp_arrow.neigh_in_port(out_port)
                neigh_arrow = neigh_port.arrow
                if neigh_arrow is not comp_arrow:
                    assert neigh_arrow in arrow_colors
                    arrow_colors[neigh_arrow] = arrow_colors[neigh_arrow] - 1
                    default_add(arrow_exprs, neigh_arrow, neigh_port.index,
                                outputs[i])

    return arrow_exprs
