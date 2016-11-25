"""Default mappings between parametric inverses and inverses"""
from reverseflow.arrows.primitive.math_arrows import *
from reverseflow.arrows.primitive.control_flow_arrows import *
from reverseflow.dispatch import *

default_dispatch = {AddArrow: inv_add,
                    SubArrow: inv_sub,
                    MulArrow: inv_mul,
                    DivArrow: inv_div,
                    DuplArrow: inv_dupl}
