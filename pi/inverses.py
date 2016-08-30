## Parametric Inverses
import tensorflow as tf

def is_constant(tensor):
    """Determine whether a tensor is constant"""
    sess = tf.Session(graph=tensor.graph)
    try:
        tensor.eval(session=sess)
    except tf.errors.InvalidArgumentError as e:
        print(type(e), e)
        return False
    return True


## How to deal with approximation
## A 'clamp' is a transformation which is applied to the input of the inverse function.
#  Which clamps its value to the domain of the inverse input
#  and outputs an error

# What about parameters
## The parameter space can be parametric with respect to the shape
##

# There are multiple different possible clamps

class Inverse:
    def __init__(self, type, param, invf):
        self.type = type
        self.param = param
        self.invf = invf

    def go(self, graph, inputs):
        # What about parameter inputs
        # What about error ouputs
        with graph.as_default():
            with graph.name_scope("inv_%s" % self.type):
                params = self.param(inputs)
                ops = self.invf(inputs, params=params)
                return ops, params

## Primitive Inverses
## ==================

## How to deal with 64 bit vs 32 bit.
## How to deal with clamping

def id(x): return tf.identity(x)

## Multiplication
def inv_mulf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_mulf(z, params): return (id(params[0]), z[0]/params[0])
invmul = Inverse('Mul', inv_mulf_param, inv_mulf)

## Addition
def inv_add_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_add(z, params): return (id(params[0]), z[0] - params[0])
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
assert typecheck_inverses(default_inverses)
