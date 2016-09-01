import tensorflow as tf
from pi.inverses import ParametricInverse, Injection
from pi.clamps import *
from pi.util import *

def inj_if_one_const(constant):
    """
    Is op with these inputs invertible
    It is if one of the arguments is constant
    """
    assert not all(constant.values()), "All inputs are constants"
    assert len(constant) == 2, "Multiplcation takes exactly two inputs"
    return any(constant.values())

inj_test = {'Mul': inj_if_one_const,
            'Add': inj_if_one_const,
            'Sub': inj_if_one_const}

## Mul
def inj_mul(inputs, fwd_inputs, constant):
    assert len(inputs) == 1, "inv_mul has one input"
    assert len(fwd_inputs) == 2, "mul has two inputs"
    z = inputs[0]
    x = fwd_inputs[0]
    y = fwd_inputs[1]
    if constant[x]:
        op = z/x
        corres = {op:y}
        return (op,), corres
    else:
        op = z/y
        corres = {op:x}
        return (op,), corres

abs_inv_mul = AbstractInverse(inj_mul)
injmul = Injection('Mul', inj_mul)

## Primitive Inverses
## ==================

## Abs
def inv_abs_param(z, intX=tf.int32): return (placeholder_like(z, dtype=intX, name="theta"),)
def inv_abs_param_real(z, floatX=tf.float32): return (placeholder_like(z, dtype=floatX, name="theta"),)
def inv_abs(z, params): return (params[0] * z[0])
# inv_abs = Inverse('Abs', inv_abs_param, inv_abs)
# inv_abs_real = Inverse('Abs', inv_abs_param_real, inv_abs)

def inv_abs_approx(z, params, clamp=lambda t: a_b_clamp(t, a=-1, b=1),
                   error=lambda t: nearest_a_b_loss(t, a=-1, b=1)):
    return inv_abs(z, (clamp(z[0]),)), error(params[0])

invabsapprox = ParametricInverse('Abs', inv_abs_param, inv_abs_approx, is_approx=True)

## Mul
def inv_mulf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_mulf(z, params): return (iden(params[0]), z[0]/params[0])
invmul = ParametricInverse('Mul', inv_mulf_param, inv_mulf, is_approx=False)

## Add
def inv_add_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_add(z, params): return (iden(params[0]), z[0] - params[0])
invadd = ParametricInverse('Add', inv_add_param, inv_add, is_approx=False)

## Sub
def inv_sub_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_sub(z, params): return (params[0] + z[0], iden(params[0]))
invsub = ParametricInverse('Sub', inv_sub_param, inv_sub, is_approx=False)

## Split
def inv_split(z): return (z[0],)
def inv_split_approx(z):
    mean = tf.add_n(z)/len(z)
    dists = [tf.abs(t - mean) for t in z]
    error = tf.add_n(dists)
    return (mean,), error

invsplit = Injection('Split', inv_split, is_approx=False)
invsplitapprox = Injection('Split', inv_split, is_approx=True)

## Sin
def inv_sinf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_sinf(z, params): return (tf.asin(z[0])*params[0],)
invsin = ParametricInverse('Sin', inv_sinf_param, inv_sinf)

## Cos
def inv_cosf_param(z): return (tf.placeholder(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_cosf(z, params): return (tf.acos(z[0])*params[0],)
invcos = ParametricInverse('Cos', inv_cosf_param, inv_cosf)
