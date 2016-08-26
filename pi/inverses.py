## Parametric Inverses
import tensorflow as tf

## How to deal with approximation
## A 'clamp' is a transformation which is applied to the input of the inverse function.
#  Which clamps its value to the domain of the inverse input
#  and outputs an error

# What about parameters
## The parameter space can be parametric with respect to the shape
##

# There are multiple different possible clamps

class Inverse:
    def __init__(self, type, invf):
        self.type = type
        self.invf = invf

    def go(self, graph, inputs):
        # What about parameter inputs
        # What about error ouputs
        with graph.as_default():
            self.invf(inputs)

## Primitive Inverses
## ==================

## Multiplication
def inv_mulf(z, theta): return (theta, z/theta)
invmul = Inverse('Mul', inv_mulf)

## Trig
def inv_sinf(z, theta): return (tf.asin(z)*theta)
invsin = Inverse('Sin', inv_sinf)

def typecheck_inverses(inverses):
    """Do types of keys in inverse list match the types of the Inverses"""
    for k,v in inverses:
        if k != v.type:
            return False

    return True
default_inverses = {'Mul', invmul,
                    'Sin', invsin}
assert typecheck_inverses(default_inverses)
