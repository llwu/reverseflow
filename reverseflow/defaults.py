"""Default mappings between parametric inverses and inverses"""
from arrows.primitive.math_arrows import *
from arrows.primitive.control_flow import *
from reverseflow.dispatch import *

default_dispatch = {AddArrow: inv_add,
                    SubArrow: inv_sub,
                    CosArrow: inv_cos,
                    DuplArrow: inv_dupl_approx,
                    ExpArrow: inv_exp,
                    GatherArrow: inv_gather,
                    GatherNdArrow: inv_gathernd,
                    MulArrow: inv_mul,
                    DivArrow: inv_div,
                    NegArrow: inv_neg,
                    BroadcastArrow: inv_broadcast,
                    ReshapeArrow: inv_reshape,
                    SinArrow: inv_sin,
                    }
