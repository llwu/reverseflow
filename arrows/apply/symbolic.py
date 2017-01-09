"""Symbolically Evaluate a Graph"""
from collections import OrderedDict
from typing import Dict, MutableMapping, Tuple, List, Set, Sequence
from arrows import Arrow, CompositeArrow, InPort, OutPort
from arrows.std_arrows import *
from reverseflow.inv_primitives.inv_math_arrows import *
from reverseflow.inv_primitives.inv_control_flow_arrows import *
from arrows.apply.interpret import interpret


from pqdict import pqdict
import sympy
from sympy import Expr, Rel, Integer, Float
from overloading import overload

ConstrainedExpr = Tuple[Expr, Set[Rel]]
ExprList = Sequence[ConstrainedExpr]


@overload
def to_sympy_number(x :int):
    return Integer(x)

@overload
def to_sympy_number(x :float):
    return Float(x)

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
def conv(arrow: CastArrow, args: ExprList) -> ExprList:
    in_args = [arg[0] for arg in args]
    total = sum(in_args)
    constraints = set()
    for arg in args:
        constraints.update(arg)
    return [(total, constraints)]

@overload
def conv(arrow: AddNArrow, args: ExprList) -> ExprList:
    return sum(args)


@overload
def conv(arrow: SourceArrow, args: ExprList) -> ExprList:
    # import pdb; pdb.set_trace()
    return [(to_sympy_number(arrow.value), set())]

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
def conv(comp_arrow: CompositeArrow, args: ExprList) -> ExprList:
    assert len(args) == comp_arrow.num_in_ports()
    return interpret(conv, comp_arrow, args)

def symbolic_apply(comp_arrow: CompositeArrow, input_exprs: List):
    """Return the output expressions and constraints of the CompositeArrow"""
    constrained_inputs = [(expr, set()) for expr in input_exprs]
    return interpret(conv, comp_arrow, constrained_inputs)
