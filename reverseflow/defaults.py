"""Default mappings between parametric inverses and inverses"""
from reverseflow.arrows.primitive.math_arrows import AddArrow, MulArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.dispatch import inv_add, inv_dupl

default_dispatch = {AddArrow: inv_add,
                    MulArrow: inv_add,
                    DuplArrow: inv_dupl} # FIXME
