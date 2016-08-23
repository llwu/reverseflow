## Parametric Inversion Test
import pi as pi
import tensorflow as tf

## How is this going to work?
class Inverse:
    def __init__(self, type, invf):
        self.type = type
        self.invf = invf

def inv_mulf(z, theta): return (theta, z/theta)
invmul = Inverse('Mul', inv_mulf)

def inv_sinf(z, theta): return (tf.asin(z)*theta)
