"""Array Operations"""
from arrows.primitivearrow import PrimitiveArrow
from arrows.port_attributes import ports_has, PortAttributes, extract_attribute
from arrows.apply.shapes import *
from arrows.apply.constants import constant_pred, constant_dispatch
import numpy as np


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
        return {constant_pred: constant_dispatch,
                gather_shape_pred: gather_shape_dispatch}


def std_pred1(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports()[1:2], 'value', port_attr)


def std_disp1(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    output_shape = const_to_tuple(port_attr[arr.in_ports()[1]]['value'])
    return {arr.out_ports()[0]: {'shape': output_shape}}


def std_pred1(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)


def std_disp1(arr: "SparseToDenseArrow", port_attr: PortAttributes):
    inds = port_attr[arr.in_ports()[0]]['value']
    output_shape = const_to_tuple(port_attr[arr.in_ports()[1]]['value'])
    vals = port_attr[arr.in_ports()[2]]['value']
    output = np.zeros(output_shape)
    for i, ind in enumerate(list(inds)):
        output[ind] = vals[i]
    return {arr.out_ports()[0]: {'value': output}}


class SparseToDenseArrow(PrimitiveArrow):
    """tf.sparse_to_dense"""
    def __init__(self):
        name = 'SparseToDense'
        super().__init__(n_in_ports=3, n_out_ports=1, name=name)

    def get_dispatches(self):
        return {constant_pred: constant_dispatch,
                std_pred1: std_disp1}

# Reshape
# ========
def reshape_eval_pred(arr: "ReshapeArrow", port_attr: PortAttributes):
    return ports_has(arr.in_ports(), 'value', port_attr)

def reshape_eval_dispatch(arr: "ReshapeArrow", port_attr: PortAttributes):
    ptv = extract_attribute('value', port_attr)
    i = arr.in_ports()
    o = arr.out_ports()
    res = np.reshape(ptv[i[0]], ptv[i[1]])
    return {o[0]: {'value': res}}

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
        return {reshape_eval_pred: reshape_eval_dispatch}
