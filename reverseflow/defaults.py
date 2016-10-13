"""Default parametric inverses to use"""

from reverseflow.inv_ops.inv_math_ops import *

def typecheck_inverses(inverses):
    """Do types of keys in inverse list match the types of the Inverses"""
    for k,v in inverses.items():
        if k != v.type:
            print("Type error ", k, v.type)
            return False

    return True


approx_inverses = {'Abs': invabsapprox,
                   'Split': invsplitapprox,
                   'Mul': invmul,
                   'Mul_Const': invmulc,
                   'Add': invadd,
                   'Add_Const': invaddc,
                   'Neg':invneg,
                   'Sub': invsub,
                   'Sub_Const1': invsubc1,
                   'Sub_Const2': invsubc2,
                   'Exp': injexp,
                   'Reshape': injreshape,
                   'Sin': injsin,
                   'Cos': injcos}

exact_inverses = {'Mul': invmul,
                    'Add': invadd,
                    'Sub': invsub,
                    'Sin': invsin,
                    'Cos': invcos,
                    'Split': invsplit}

assert typecheck_inverses(exact_inverses)
assert typecheck_inverses(approx_inverses)
default_inverses = approx_inverses
dispatches = {'Mul':dispatch_mul,
              'Add':dispatch_add,
              'Sub':dispatch_sub,
              'Neg':dispatch_neg,
              'Sin':dispatch_sin,
              'Cos':dispatch_cos,
              'Reshape':dispatch_reshape,
              'Exp':dispatch_exp}
