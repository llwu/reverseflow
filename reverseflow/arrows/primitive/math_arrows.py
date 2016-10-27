from reverseflow.arrows.primitivearrow import PrimitiveArrow


class AddArrow(PrimitiveArrow):
    """Addition"""

    def __init__(self):
        name = 'Add'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)

class SubArrow(PrimitiveArrow):
    """Subtraction. Out[1] = In[0] - In[1]"""

    def __init__(self):
        name = 'Sub'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)

class MulArrow(PrimitiveArrow):
    """Multiplication"""

    def __init__(self):
        name = 'Mul'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)

class DivArrow(PrimitiveArrow):
    """Division"""

    def __init__(self):
        name = 'Div'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)

class ExpArrow(PrimitiveArrow):
    """Exponentiaion"""

    def __init__(self):
        name = 'Exp'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)

class LogArrow(PrimitiveArrow):
    """Logarithm"""

    def __init__(self):
        name = 'Log'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)

class NegArrow(PrimitiveArrow):
    """Negation"""

    def __init__(self):
        name = 'Neg'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def gen_constraints(self, input_expr: Dict[Int, Expr], output_expr: Dict[Int, Expr]) -> List[Expr]:
        super().gen_constraints(input_expr, output_expr)
