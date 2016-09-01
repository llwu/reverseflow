from pi.inv_ops.inv_math_ops import *

def typecheck_inverses(inverses):
    """Do types of keys in inverse list match the types of the Inverses"""
    for k,v in inverses.items():
        if k != v.type:
            return False

    return True


approx_inverses = {'Abs': invabsapprox,
                   'Split': invsplitapprox,
                   'Mul': invmul,
                   'Add': invadd,
                   'Sub': invsub}

exact_inverses = {'Mul': invmul,
                    'Add': invadd,
                    'Sub': invsub,
                    'Sin': invsin,
                    'Cos': invcos,
                    'Split': invsplit}

assert typecheck_inverses(exact_inverses)
assert typecheck_inverses(approx_inverses)
default_inverses = approx_inverses
