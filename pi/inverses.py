## Parametric Inverses
import tensorflow as tf

## Two things to deal with constants
##


## How to deal with approximation
## A 'clamp' is a transformation which is applied to the input of the inverse function.
#  Which clamps its value to the domain of the inverse input
#  and outputs an error

# What about parameters
## The parameter space can be parametric with respect to the shape
##

# There are multiple different possible clamps

class Inverse:
    """Parametric Inverse"""
    def __init__(self, type, param, invf):
        self.type = type
        self.param = param
        self.invf = invf

    def go(self, graph, inputs):
        print("PINV INP", inputs)
        # What about error ouputs
        with graph.as_default():
            with graph.name_scope("inv_%s" % self.type):
                params = self.param(inputs)
                ops = self.invf(inputs, params=params)
                return ops, params

class Injection:
    """Invertible (Injective) Fucntion"""
    def __init__(self, type, invf):
        self.type = type
        self.invf = invf

    def go(self, graph, inputs, fwd_inputs, constants):
        print("INJ INP", inputs)
        # What about error ouputs
        with graph.as_default():
            with graph.name_scope("inj_%s" % self.type):
                ops, corres = self.invf(inputs, fwd_inputs, constants)
                return ops, corres

# def copy_tensor(tensor):

## Is op with these inputs invertible
## It is if one of the arguments is constant
def inj_if_one_const(constant):
    assert not all(constant.values()), "All inputs are constants"
    assert len(constant) == 2, "Multiplcation takes exactly two inputs"
    return any(constant.values())

inj_test = {'Mul': inj_if_one_const,
            'Add': inj_if_one_const}
## Invertible Functions
## ====================

## Arithmetic by constant
## Multiplication
def inj_mul(inputs, fwd_inputs, constant):
    assert len(inputs) == 1, "inv_mul has one input"
    assert len(fwd_inputs) == 2, "mul has two inputs"
    z = inputs[0]
    x = fwd_inputs[0]
    y = fwd_inputs[1]

    ## Need to build correspodnance between output of inv function
    ## and input of fwd function
    print("Z",z, "x",x, "y",y)
    if constant[x]:
        op = z/x
        corres = {op:y}
        return (op,), corres
    else:
        op = z/y
        corres = {op:x}
        return (op,), corres

injmul = Injection('Mul', inj_mul)
default_injections = {'Mul': injmul}



## Primitive Inverses
## ==================

## How to deal with 64 bit vs 32 bit.
## How to deal  with clamping

iden = tf.identity

def inv_mulf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_mulf(z, params): return (iden(params[0]), z[0]/params[0])
invmul = Inverse('Mul', inv_mulf_param, inv_mulf)

## Addition
def inv_add_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_add(z, params): return (iden(params[0]), z[0] - params[0])
invadd = Inverse('Add', inv_add_param, inv_add)

## Split
def inv_split_param(z): return ()
def inv_split(z, params): print("SHITFACE", z); return (z[0],)
invsplit = Inverse('Split', inv_split_param, inv_split)

## Trig
def inv_sinf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_sinf(z, params): return (tf.asin(z[0])*params[0],)
invsin = Inverse('Sin', inv_sinf_param, inv_sinf)

## Trig
def inv_cosf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_cosf(z, params): return (tf.acos(z[0])*params[0],)
invcos = Inverse('Cos', inv_cosf_param, inv_cosf)

def typecheck_inverses(inverses):
    """Do types of keys in inverse list match the types of the Inverses"""
    for k,v in inverses.items():
        if k != v.type:
            return False

    return True
default_inverses = {'Mul': invmul,
                    'Add': invadd,
                    'Sin': invsin,
                    'Cos': invcos,
                    'Split': invsplit}

## Type Checking
## =============
assert typecheck_inverses(default_inverses)
assert typecheck_inverses(default_injections)
