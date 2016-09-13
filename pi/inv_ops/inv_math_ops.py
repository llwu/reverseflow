import tensorflow as tf
from pi.inverses import ParametricInverse, Injection
from pi.clamps import *
from pi.util import *
import pdb

def evaly(t):
    sess = tf.Session(graph=t.graph)
    res = t.eval(session=sess)
    sess.close()
    return res

def dispatch_reshape(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    concrete_shape = fwd_inputs[0].get_shape()
    op = inverses['Reshape'].apply(graph, inv_inputs, shape=concrete_shape)
    corres = {op[0]: fwd_inputs[0]}
    return op, corres

def inj_reshapef(z, shape): return (tf.reshape(z[0], shape),)
injreshape = Injection('Reshape', inj_reshapef, is_approx=False)

def dispatch_exp(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    # pdb.set_trace()
    assert len(inv_inputs) == 1, "inv_exp has one input"
    assert len(fwd_inputs) == 1, "exp has one inputs"
    op = inverses['Exp'].apply(graph, inv_inputs)
    corres = {op[0]:fwd_inputs[0]}
    return op, corres

def inj_expf(z): return (tf.log(z[0]),)
injexp = Injection('Exp', inj_expf, is_approx=False)

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
def dispatch_mul(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    assert len(inv_inputs) == 1, "inv_mul has one input"
    assert len(fwd_inputs) == 2, "mul has two inputs"
    constant = {fwd_inp:is_constant(fwd_inp) for fwd_inp in fwd_inputs}
    x = fwd_inputs[0]
    y = fwd_inputs[1]
    assert not (constant[x] and constant[y]), "Both inputs constant"
    if constant[x]:
        op = inverses['Mul_Const'].apply(graph, inv_inputs, consts=(x,))
        corres = {op[0]:y}
        return op, corres
    elif constant[y]:
        op = inverses['Mul_Const'].apply(graph, inv_inputs, consts=(y,))
        corres = {op[0]:x}
        return op, corres
    else:
        op = inverses['Mul'].apply(graph, inv_inputs, shrunk_params=shrunk_params)
        assert len(op) == len(fwd_inputs)
        corres = {op[i]:fwd_inputs[i] for i in range(len(op))}
        return op, corres

def dispatch_add(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    assert len(inv_inputs) == 1, "inv_add has one input"
    assert len(fwd_inputs) == 2, "add has two inputs"
    constant = {fwd_inp:is_constant(fwd_inp) for fwd_inp in fwd_inputs}
    x = fwd_inputs[0]
    y = fwd_inputs[1]
    assert not (constant[x] and constant[y]), "Both inputs constant"
    if constant[x]:
        op = inverses['Add_Const'].apply(graph, inv_inputs, consts=(x,))
        corres = {op[0]:y}
        return op, corres
    elif constant[y]:
        op = inverses['Add_Const'].apply(graph, inv_inputs, consts=(y,))
        corres = {op[0]:x}
        return op, corres
    else:
        op = inverses['Add'].apply(graph, inv_inputs, shrunk_params=shrunk_params)
        assert len(op) == len(fwd_inputs)
        corres = {op[i]:fwd_inputs[i] for i in range(len(op))}
        return op, corres

def dispatch_sub(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    assert len(inv_inputs) == 1, "inv_sub has one input"
    assert len(fwd_inputs) == 2, "sub has two inputs"
    constant = {fwd_inp:is_constant(fwd_inp) for fwd_inp in fwd_inputs}
    x = fwd_inputs[0]
    y = fwd_inputs[1]
    assert not (constant[x] and constant[y]), "Both inputs constant"
    if constant[x]:
        op = inverses['Sub_Const1'].apply(graph, inv_inputs, consts=(x,))
        corres = {op[0]:y}
        return op, corres
    elif constant[y]:
        op = inverses['Sub_Const2'].apply(graph, inv_inputs, consts=(y,))
        corres = {op[0]:x}
        return op, corres
    else:
        op = inverses['Sub'].apply(graph, inv_inputs, shrunk_params=shrunk_params)
        assert len(op) == len(fwd_inputs)
        corres = {op[i]:fwd_inputs[i] for i in range(len(op))}
        return op, corres

def dispatch_neg(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    assert len(inv_inputs) == 1, "inv_sub has one input"
    assert len(fwd_inputs) == 1, "sub has one inputs"
    op = inverses['Neg'].apply(graph, inv_inputs)
    corres = {op[0]:fwd_inputs[0]}
    return op, corres

def dispatch_cos(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    assert len(inv_inputs) == 1, "inv_sub has one input"
    assert len(fwd_inputs) == 1, "sub has one inputs"
    op = inverses['Cos'].apply(graph, inv_inputs)
    corres = {op[0]:fwd_inputs[0]}
    return op, corres

def dispatch_sin(graph, inv_inputs, fwd_inputs, shrunk_params, inverses):
    assert len(inv_inputs) == 1, "inv_sub has one input"
    assert len(fwd_inputs) == 1, "sub has one inputs"
    op = inverses['Sin'].apply(graph, inv_inputs)
    corres = {op[0]:fwd_inputs[0]}
    return op, corres

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
def inv_mulf_param(z): return (tensor_type(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_mulf(z, params): return (iden(params[0]), z[0]/params[0])
# def inv_mulf(z, params): return (tf.pow(z[0], params[0]), tf.pow(z[0], 1-params[0]))
invmul = ParametricInverse('Mul', inv_mulf, inv_mulf_param, is_approx=False)

def inv_mulc(z, consts): return (z[0]/ consts[0],)
invmulc = Injection('Mul_Const', inv_mulc, is_approx=False)

## Add
def inv_add_param(z): return (tensor_type(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_add(z, params): return (iden(params[0]), z[0] - params[0])
invadd = ParametricInverse('Add', inv_add, inv_add_param, is_approx=False)

def inv_addc(z, consts): return (z[0] - consts[0],)
invaddc = Injection('Add_Const', inv_addc, is_approx=False)

## Sub
def inv_sub_param(z): return (tensor_type(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_sub(z, params): return (params[0] + z[0], iden(params[0]))
invsub = ParametricInverse('Sub', inv_sub, inv_sub_param, is_approx=False)

def inv_subc1(z, consts): return (consts[0] - z[0],)
invsubc1 = Injection('Sub_Const1', inv_subc1, is_approx=False)

def inv_subc2(z, consts): return (consts[0] + z[0],)
invsubc2 = Injection('Sub_Const2', inv_subc2, is_approx=False)

## Neg
def inv_neg(z): return (-z[0],)
invneg = Injection("Neg", inv_neg, is_approx=False)

## Split
def inv_split(z): return (z[0],)
def inv_split_approx(z):
    """Outputs the mean of its inputs and the error as the variance"""
    mean = tf.add_n(z)/len(z)
    variances = [tf.abs(t - mean) for t in z]
    mean_variances = tf.add_n(variances)/len(z)
    batched_error = tf.reduce_mean(mean_variances,
                                   reduction_indices=dims_bar_batch(mean_variances))
    return (mean,), (batched_error,)

invsplit = Injection('Split', inv_split, is_approx=False)
invsplitapprox = Injection('Split', inv_split_approx, is_approx=True)



def bound_loss(x, a, b):
    a = tf.constant(a, shape=x.get_shape())
    b = tf.constant(b, shape=x.get_shape())
    zero = tf.constant(0.0, shape=x.get_shape())
    print((x>b).get_shape())
    print("s@",(x-b).get_shape())
    g = tf.select(x>b, x-b, tf.select(x<a, a-x, zero))
    return tf.reduce_mean(g,reduction_indices=dims_bar_batch(g))

## Sin
def inv_sinf_param(z): return (tensor_type(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_sinf(z, params): return (tf.asin(z[0])*params[0],)
invsin = ParametricInverse('Sin', inv_sinf_param, inv_sinf, is_approx=True)

def inj_sinf(z): return (tf.asin(tf.clip_by_value(z[0],-1,1)),), (bound_loss(z[0], -1.0, 1.0),)
injsin = Injection('Sin', inj_sinf, is_approx=True)

## Cos
def inv_cosf_param(z): return (tensor_type(z[0].dtype, shape=z[0].get_shape(), name="theta"),)
def inv_cosf(z, params): return (tf.acos(z[0])*params[0],)
invcos = ParametricInverse('Cos', inv_cosf_param, inv_cosf, is_approx=True)

def inj_cosf(z): return (tf.acos(tf.clip_by_value(z[0],-1,1)),), (bound_loss(z[0], -1.0, 1.0),)
injcos = Injection('Cos', inj_cosf, is_approx=True)
