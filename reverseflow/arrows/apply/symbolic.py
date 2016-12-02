"""Symbolically Evaluate a Graph"""
from collections import OrderedDict
from typing import Dict, MutableMapping, Tuple, List, Set
from reverseflow.arrows.port import InPort, OutPort
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.primitive.math_arrows import (AddArrow,
    MulArrow)
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.inv_primitives.inv_math_arrows import (InvAddArrow,
    InvMulArrow)
from reverseflow.inv_primitives.inv_control_flow_arrows import InvDuplArrow


from pqdict import pqdict
import sympy
from sympy import Expr, Rel
from overloading import overload

ConstrainedExpr = Tuple[Expr, Set[Rel]]
ExprList = List[ConstrainedExpr]

def get_constraints(arrow: Arrow, in_args: List[Expr], out_args: List[Expr], new_var = False) -> Set[Rel]:
    constraints = set()
    if new_var == True:
        new_out_args = []
        for i, out in enumerate(out_args):
            out_var = sympy.var(arrow.name + '_out_' + str(i))
            # constraints.add(Eq(out_var, out))
            new_out_args.append(out_var)
        out_args = new_out_args[:]
    input_expr = OrderedDict()  # type: MutableMapping[int, Expr]
    output_expr = OrderedDict()  # type: MutableMapping[int, Expr]
    for i, in_expr in enumerate(in_args):
        input_expr[i] = in_expr
    for i, out_expr in enumerate(out_args):
        output_expr[i] = out_expr
    constraints.update(generate_constraints(arrow, input_expr, output_expr))
    return constraints

def generate_constraints(arrow: Arrow,
                         input_expr: MutableMapping[int, Expr],
                         output_expr: MutableMapping[int, Expr]) -> Set[Rel]:
    if arrow.is_primitive():
        return arrow.gen_constraints(input_expr, output_expr)

@overload
def conv(add: AddArrow, args: ExprList) -> ExprList:
    assert len(args) == 2
    in_args = [arg[0] for arg in args]
    out_args = [in_args[0] + in_args[1]]
    constraints = get_constraints(add, in_args, out_args)
    for arg in args:
        constraints.update(arg[1])
    output = [(out_arg, constraints) for out_arg in out_args]
    return output

@overload
def conv(mul: MulArrow, args: ExprList) -> ExprList:
    assert len(args) == 2
    in_args = [arg[0] for arg in args]
    out_args = [in_args[0] * in_args[1]]
    constraints = get_constraints(mul, in_args, out_args)
    for arg in args:
        constraints.update(arg[1])
    output = [(out_arg, constraints) for out_arg in out_args]
    return output

@overload
def conv(sub: SubArrow, args: ExprList) -> ExprList:
    assert len(args) == 2
    in_args = [arg[0] for arg in args]
    out_args = [in_args[0] - in_args[1]]
    constraints = get_constraints(sub, in_args, out_args)
    for arg in args:
        constraints.update(arg[1])
    output = [(out_arg, constraints) for out_arg in out_args]
    return output

@overload
def conv(div: DivArrow, args: ExprList) -> ExprList:
    assert len(args) == 2
    in_args = [arg[0] for arg in args]
    out_args = [in_args[0] / in_args[1]]
    constraints = get_constraints(div, in_args, out_args)
    for arg in args:
        constraints.update(arg[1])
    output = [(out_arg, constraints) for out_arg in out_args]
    return output

@overload
def conv(dupl: DuplArrow, args: ExprList) -> ExprList:
    assert len(args) == 1
    in_args = [arg[0] for arg in args]
    out_args = [in_args[0] for i in range(dupl.n_out_ports)]
    constraints = get_constraints(dupl, in_args, out_args)
    for arg in args:
        constraints.update(arg[1])
    output = [(out_arg, constraints) for out_arg in out_args]
    return output

@overload
def conv(inv_dupl: InvDuplArrow, args: ExprList) -> ExprList:
    assert len(args) == inv_dupl.num_in_ports()
    in_args = [arg[0] for arg in args]
    out_args = [in_args[0]]
    constraints = get_constraints(inv_div, in_args, out_args, new_var = True)
    for arg in args:
        constraints.update(arg[1])
    output = [(out_arg, constraints) for out_arg in out_args]
    return output

@overload
def conv(comp_arrow: CompositeArrow, input_args: ExprList) -> ExprList:
    assert len(args) == comp_arrow.num_in_ports()
    arrow_colors, arrow_tensors = inner_convert(comp_arrow, input_args)
    result = arrow_to_graph(conv,
                            comp_arrow,
                            input_args,
                            arrow_colors,
                            arrow_tensors)
    return result['output_tensors']


def symbolic_apply(comp_arrow: CompositeArrow):
    """Return the output expressions and constraints of the CompositeArrow"""
    input_exprs = []
    for i in range(comp_arrow.num_in_ports()):
        in_expr = sympy.var(comp_arrow.name + '_inp_' + str(i))
        input_expr.append((in_expr, set()))
    return interpret(conv, comp_arrow, input_exprs)
