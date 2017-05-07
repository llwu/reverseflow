"""Array Operations"""
import numpy as np
import tensorflow as tf

import arrows.compositearrow as compositearrows
from arrows.primitivearrow import PrimitiveArrow
from arrows.primitive.math_arrows import AddArrow
from arrows.port_attributes import ports_has, PortAttributes, extract_attribute
from arrows.apply.shapes import *
from arrows.apply.constants import constant_pred, constant_dispatch
from reverseflow.util.mapping import Bimap
from reverseflow.util.misc import complement_bool


def const_to_tuple(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return (x.item(),)
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return tuple(x)
    if not isinstance(x, tuple):
        return (x,)
    return x


def gather_shape_pred(arr: "GatherArrow", port_attr: PortAttributes):
    # FIXME: Can we infer shaep from output or aoutput and one input?
    return ports_has(arr.in_ports(), 'shape', port_attr)


def gather_shape_dispatch(arr: "GatherArrow", port_attr: PortAttributes):
    # Produces an output tensor with shape `indices.shape + params.shape[1:]`
    pts = extract_attribute('shape', port_attr)
    indices_shape = const_to_tuple(pts[arr.in_ports()[1]])
    param_shape = const_to_tuple(pts[arr.in_ports()[0]])
    return {arr.out_ports()[0]: {'shape': indices_shape + param_shape[1:]}}

class GatherArrow(PrimitiveArrow):
    """
    Gather slices from `params` according to `indices`.

    `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    Produces an output tensor with shape `indices.shape + params.shape[1:]`,
    where:

    for i = 0...rank(indices)
    output[i] = params[indices[i]]



    ```python
        # Scalar indices
        output[:, ..., :] = params[indices, :, ... :]

        # Vector indices
        output[i, :, ..., :] = params[indices[i], :, ... :]

        # Higher rank indices
        output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
    ```

    If `indices` is a permutation and `len(indices) == params.shape[0]` then
    this operation will permute `params` accordingly.

    Args:
      params: A `Tensor`.
      indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      validate_indices: An optional `bool`. Defaults to `True`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `params`.

    """

    def __init__(self):
        name = 'Gather'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({gather_shape_pred: gather_shape_dispatch})
        return disp


def gathernd_shape_pred(arr: "GatherArrow", port_attr: PortAttributes):
    # FIXME: Can we infer shaep from output or aoutput and one input?
    return ports_has(arr.in_ports(), 'shape', port_attr)


def gathernd_shape_dispatch(arr: "GatherArrow", port_attr: PortAttributes):
    # [d_0, ..., d_{Q-2}, params.shape[K], ..., params.shape[P-1]].
    pts = extract_attribute('shape', port_attr)
    indices_shape = const_to_tuple(pts[arr.in_ports()[1]])
    param_shape = const_to_tuple(pts[arr.in_ports()[0]])
    return {arr.out_ports()[0]: {'shape': indices_shape[:-1] + param_shape[indices_shape[-1]:]}}


class GatherNdArrow(PrimitiveArrow):
    """
    tf.gather_nd
    """

    def __init__(self):
        name = 'GatherNd'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({gathernd_shape_pred: gathernd_shape_dispatch})
        return disp


def std_pred1(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports()[1:2], 'value', port_attr)


def std_disp1(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    output_shape = const_to_tuple(port_attr[arr.in_ports()[1]]['value'])
    return {arr.out_ports()[0]: {'shape': output_shape}}


def std_pred2(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)


def std_disp2(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    inds = port_attr[arr.in_ports()[0]]['value']
    output_shape = const_to_tuple(port_attr[arr.in_ports()[1]]['value'])
    vals = port_attr[arr.in_ports()[2]]['value']
    output = np.zeros(output_shape)
    for i, ind in enumerate(list(inds)):
        output[ind] = vals[i]
    return {arr.out_ports()[0]: {'value': output}}


def std_pred3(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports()[0:1], 'shape', port_attr)


def std_disp3(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    inds_shape = port_attr[arr.in_ports()[0]]['shape']
    return {arr.in_ports()[2]: {'shape': (inds_shape[0],)}}

def std_pred4(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    return ports_has(arr.out_ports(), 'shape', port_attr)


def std_disp4(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    output_shape = const_to_tuple(port_attr[arr.out_ports()[0]]['shape'])
    return {arr.in_ports()[1]: {'value': output_shape}}

# For symbolic tensor
from arrows.util.tf import tf_eval
from arrows.transform.symbolic_tensor import SymbolicTensor
import tensorflow as tf

def std_symbt_pred(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    # sparse_indices: value
    # output_shape: value
    # sparse_values: SymbolicTensor
    a = ports_has(arr.in_ports()[0:2], 'value', port_attr)
    b = port_has(arr.in_port(2), 'symbolic_tensor', port_attr)
    # Hack, a FIXME to propagate should stop repropagation
    c = not port_has(arr.out_port(0), 'symbolic_tensor', port_attr)
    return a and b and c


def std_symb_disp(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    st = port_attr[arr.in_ports()[2]]['symbolic_tensor']
    indices = port_attr[arr.in_ports()[0]]['value']
    output_shape = port_attr[arr.in_ports()[1]]['value']
    values = st.indices
    res = tf_eval(tf.sparse_to_dense, sparse_indices=indices, sparse_values=values, output_shape=output_shape)
    st = SymbolicTensor(indices=res, symbols=st.symbols, name=st.name, port=st.port)
    return {arr.out_port(0): {'symbolic_tensor': st}}



class SparseToDenseArrow(PrimitiveArrow):
    """tf.sparse_to_dense"""
    def __init__(self):
        name = 'SparseToDense'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            std_pred1: std_disp1,
            std_pred2: std_disp2,
            std_pred3: std_disp3,
            std_pred4: std_disp4,
            std_symbt_pred: std_symb_disp,
            })
        return disp


def snd_pred1(arr: "ScatterNdArrow", port_attr: PortAttributes):
    return port_has(arr.in_ports()[2], 'value', port_attr)


def snd_disp1(arr: "ScatterNdArrow", port_attr: PortAttributes):
    output_shape = const_to_tuple(port_attr[arr.in_port(2)]['value'])
    return {arr.out_ports()[0]: {'shape': output_shape}}


def snd_pred2(arr: "ScatterNdArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)


def snd_disp2(arr: "ScatterNdArrow", port_attr: PortAttributes):
    inds = tf.constant(port_attr[arr.in_port(0)]['value'])
    vals = tf.constant(port_attr[arr.in_port(1)]['value'])
    output_shape = tf.constant(port_attr[arr.in_port(2)]['value'])
    scatter = tf.scatter_nd(inds, vals, output_shape)
    with tf.Session() as sess:
        return {arr.out_port(0): {'value': sess.run(scatter)}}


def snd_pred3(arr: "ScatterNdArrow", port_attr: PortAttributes):
    return port_has(arr.in_port(0), 'shape', port_attr) and port_has(arr.in_port(2), 'value', port_attr)


def snd_disp3(arr: "ScatterNdArrow", port_attr: PortAttributes):
    # [d_0, ..., d_{Q-2}, params.shape[K], ..., params.shape[P-1]].
    indices_shape = const_to_tuple(port_attr[arr.in_port(0)]['shape'])
    params_shape = const_to_tuple(port_attr[arr.in_port(2)]['value'])
    return {arr.in_ports()[1]: {'shape': indices_shape[:-1] + params_shape[indices_shape[-1]:]}}

def snd_pred4(arr: "ScatterNdArrow", port_attr: PortAttributes):
    return ports_has(arr.out_ports(), 'shape', port_attr)


def snd_disp4(arr: "ScatterNdArrow", port_attr: PortAttributes):
    output_shape = np.array(port_attr[arr.out_ports()[0]]['shape'])
    return {arr.in_ports()[2]: {'value': output_shape}}

def snd_pred5(arr: "ScatterNdArrow", port_attr: PortAttributes):
    return port_has(arr.in_port(0), 'shape', port_attr) and port_has(arr.out_port(0), 'shape', port_attr)


def snd_disp5(arr: "ScatterNdArrow", port_attr: PortAttributes):
    # [d_0, ..., d_{Q-2}, params.shape[K], ..., params.shape[P-1]].
    indices_shape = const_to_tuple(port_attr[arr.in_port(0)]['shape'])
    params_shape = const_to_tuple(port_attr[arr.out_port(0)]['shape'])
    return {arr.in_ports()[1]: {'shape': indices_shape[:-1] + params_shape[indices_shape[-1]:]}}

def snd_pred6(arr: "ScatterNdArrow", port_attr: PortAttributes):
    return port_has(arr.in_port(0), 'value', port_attr) and port_has(arr.out_port(0), 'value', port_attr)


def snd_disp6(arr: "ScatterNdArrow", port_attr: PortAttributes):
    inds = tf.constant(port_attr[arr.in_port(0)]['value'])
    output = tf.constant(port_attr[arr.out_port(0)]['value'])
    gather_nd = tf.gather_nd(output, inds)
    with tf.Session() as sess:
        return {arr.in_ports()[1]: {'value': sess.run(gather_nd)}}

def snd_symbt_pred(arr: "ScatterNdArrow", port_attr: PortAttributes):
    # sparse_indices: value
    # output_shape: value
    # sparse_values: SymbolicTensor
    a = port_has(arr.in_port(0), 'value', port_attr)
    b = port_has(arr.in_port(1), 'symbolic_tensor', port_attr)
    # Hack, a FIXME to propagate should stop repropagation
    c = not port_has(arr.out_port(0), 'symbolic_tensor', port_attr)
    d = port_has(arr.in_port(2), 'value', port_attr)
    return a and b and c and d


def snd_symb_disp(arr: "ScatterNdArrow", port_attr: PortAttributes):
    st = port_attr[arr.in_ports()[1]]['symbolic_tensor']
    indices = port_attr[arr.in_ports()[0]]['value']
    output_shape = port_attr[arr.in_ports()[2]]['value']
    values = st.indices
    res = tf_eval(tf.scatter_nd, indices=indices, updates=values, shape=output_shape)
    st = SymbolicTensor(indices=res, symbols=st.symbols, name=st.name, port=st.port)
    return {arr.out_port(0): {'symbolic_tensor': st}}



class ScatterNdArrow(PrimitiveArrow):
    """tf.scatter_nd"""
    def __init__(self):
        name = 'ScatterNd'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            snd_pred1: snd_disp1,
            snd_pred2: snd_disp2,
            snd_pred3: snd_disp3,
            snd_pred4: snd_disp4,
            snd_pred5: snd_disp5,
            snd_pred6: snd_disp6,
            snd_symbt_pred: snd_symb_disp,
            })
        return disp

# Reshape
# FIXME: We don't consider the special value -1, which is supposed to infer shape
# ========
def reshape_eval_pred(arr: "ReshapeArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def reshape_eval_dispatch(arr: "ReshapeArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    res = np.reshape(ptv[i[0]], ptv[i[1]])
    return {o[0]: {'value': res}}

def reshape_pred1(arr: "ReshapeArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports()[:1], 'shape', port_attr) and ports_has(arr.out_ports()[:1], 'value', port_attr)

def reshape_dispatch1(arr: "ReshapeArrow", port_attr: PortAttributes):
    o = port_attr[arr.out_ports()[0]]['value']
    s = port_attr[arr.in_ports()[0]]['shape']
    return {arr.in_ports()[0]: {'value': np.reshape(o, const_to_tuple(s))}}

def reshape_pred2(arr: "ReshapeArrow", port_attr: PortAttributes):
    return ports_has(arr.out_ports()[:1], 'shape', port_attr)

def reshape_dispatch2(arr: "ReshapeArrow", port_attr: PortAttributes):
    o = port_attr[arr.out_ports()[0]]['shape']
    return {arr.in_ports()[1]: {'value': np.array(o)}}

def reshape_pred3(arr: "ReshapeArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports()[1:2], 'value', port_attr)

def reshape_dispatch3(arr: "ReshapeArrow", port_attr: PortAttributes):
    i = port_attr[arr.in_ports()[1]]['value']
    return {arr.out_ports()[0]: {'shape': const_to_tuple(i)}}

class ReshapeArrow(PrimitiveArrow):
    """
    Port0:  Tensor
    Port1:  Shape
    Port1: Reshaped Tensor
    """

    def __init__(self):
        name = 'Reshape'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({reshape_eval_pred: reshape_eval_dispatch,
            reshape_pred1: reshape_dispatch1,
            reshape_pred2: reshape_dispatch2,
            reshape_pred3: reshape_dispatch3
            })
        return disp


class SliceArrow(PrimitiveArrow):
    """
    tf.slice
    """
    def __init__(self):
        name = 'Slice'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)


class SqueezeArrow(PrimitiveArrow):
    """
    tf.squeeze
    """
    def __init__(self):
        name = 'Squeeze'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)


def fwd_pred(arr: "SelectArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def fwd_disp(arr: "SelectArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    return {o[0] : {'value': np.where(ptv[i[0]], ptv[i[1]], ptv[i[2]])}}

def bwd_pred(arr: "SelectArrow", port_attr: PortAttributes):
    return arr.zeros and port_has(arr.out_port(0), 'value', port_attr) and port_has(arr.in_port(0), 'value', port_attr)

def bwd_disp(arr: "SelectArrow", port_attr: PortAttributes):
    indices = port_attr[arr.in_port(0)]['value']
    out_val = port_attr[arr.out_port(0)]['value']
    vals = {}
    vals[arr.in_port(1)] = {'value': np.where(indices, out_val, 0)}
    vals[arr.in_port(2)] = {'value': np.where(indices, 0, out_val)}
    return vals


class SelectArrow(PrimitiveArrow):
    """
    tf.select
    """
    def __init__(self, zeros=False):
        """zeros is whether the nonselected elements are already zero. For propagation"""
        name = 'Select'
        self.zeros = zeros
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            fwd_pred: fwd_disp,
            bwd_pred: bwd_disp
            })
        return disp


def zeros_pred(arr: "UpdateArrow", port_attr: PortAttributes):
    return port_has(arr.out_port(0), 'value', port_attr) and port_has(arr.in_port(1), 'value', port_attr) and port_has(arr.in_port(3), 'value', port_attr)

def zeros_disp(arr: "UpdateArrow", port_attr: PortAttributes):
    inds = port_attr[arr.in_port(1)]['value']
    shape = port_attr[arr.in_port(3)]['value']
    bools = complement_bool(inds, shape)
    output = port_attr[arr.out_port(0)]['value']
    return {arr.in_ports()[0]: {'value': np.where(bools, output, 0)}}

class UpdateArrow(compositearrows.CompositeArrow):
    """
    Not a tensorflow op. Just wraps scatter and exclusive add so we can propagate
    """

    def __init__(self):
        name = "Update"
        edges = Bimap()
        scatter = ScatterNdArrow()
        add = AddArrow()
        edges.add(scatter.out_port(0), add.in_port(1))  # must be such that we are only adding these elements to zeros
        super().__init__(edges=edges,
                         in_ports=add.in_ports()[:1]+scatter.in_ports(),
                         out_ports=add.out_ports(),
                         name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            zeros_pred: zeros_disp
            })
        return disp


def stack_shapes(shapes: Sequence, axis: int):
    """if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
       if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
    """
    assert same(shapes), "shapes of input to stack should be the same"
    s = shapes[0]
    n = len(shapes)
    new_shape = s[0:axis] + [n] + s[axis:]
    return tuple(new_shape)


def stack_shape_pred(arr: "StackArrow", port_attr: PortAttributes):
    q = ports_has(arr.in_ports(), 'shape', port_attr)
    return ports_has(arr.in_ports(), 'shape', port_attr)


def stack_shape_disp(arr: "StackArrow", port_attr: PortAttributes):
    ptv = extract_attribute('shape', port_attr)
    import pdb; pdb.set_trace()
    o = port_attr[arr.out_port(0)]
    return {arr.out_port(0): {'shape': new_shape}}


class StackArrow(PrimitiveArrow):
    """
    tf.stack
    """
    def __init__(self, n_inputs, axis):
        """Stacks inputs along axis"""
        name = 'Stack'
        self.axis = axis
        super().__init__(n_in_ports=n_inputs, n_out_ports=1, name=name)

    def get_dispatches(self):
          disp = super().get_dispatches()
          disp.update({
              stack_shape_pred: stack_shape_disp
              })
          return disp


class TransposeArrow(PrimitiveArrow):
    """
    tf.stack
    """
    def __init__(self, perm):
        """Stacks inputs along perm"""
        name = 'Transpose'
        self.perm = perm
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            })
        return disp
