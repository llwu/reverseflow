from typing import List

import numpy as np
import tensorflow as tf

from arrows.apply.propagate import propagate
from arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_graph
from arrows.config import floatX
from arrows.port_attributes import (is_error_port, extract_attribute,
  is_param_port, get_port_dtype)


def apply(arrow: Arrow, inputs: List[np.ndarray]) -> List[np.ndarray]:
    """Apply an arrow to some inputs.  Uses tensorflow for actual computation.
    Args:
        Arrow: The Arrow to compute
        inputs: Input values to the arrow
    Returns:
        list of outputs Arrow(inputs)"""
    assert len(inputs) == arrow.num_in_ports(), "wrong # inputs"
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        input_tensors = []
        for i in range(len(inputs)):
            dtype = get_port_dtype(arrow.in_port(i))
            input_tensors.append(tf.placeholder(dtype=dtype,
                                 shape=inputs[i].shape))
        outputs = arrow_to_graph(arrow, input_tensors)
        feed_dict = dict(zip(input_tensors, inputs))
        init = tf.global_variables_initializer()
        sess.run(init)
        outputs = sess.run(fetches=outputs,
                           feed_dict=feed_dict)
    sess.close()
    return outputs


def apply_backwards(arrow: Arrow, outputs: List[np.ndarray], port_attr=None) -> List[np.ndarray]:
    """
    Takes out_port vals (excluding errors) and returns in_port vals (including params).
    FIXME: Mutates port_attr
    """
    out_ports = [out_port for out_port in arrow.out_ports() if not is_error_port(out_port)]
    if port_attr is None:
        port_attr = propagate(arrow)
    for i, out_port in enumerate(out_ports):
        if out_port not in port_attr:
            port_attr[out_port] = {}
        port_attr[out_port]['value'] = outputs[i]
    for out_port in arrow.out_ports():
        if is_error_port(out_port):
            if out_port not in port_attr:
                port_attr[out_port] = {}
            if 'shape' in port_attr[out_port]:
                port_attr[out_port]['value'] =  np.zeros(port_attr[out_port]['shape'])
            else:
                # FIXME: there has to be a better way to do this
                print("WARNING: shape of error port unknown: %s" % (out_port))
                port_attr[out_port]['value'] = 0

    port_attr = propagate(arrow, port_attr)#, only_prop=set(['value']))
    vals = extract_attribute('value', port_attr)
    in_vals = {port: vals[port] for port in arrow.in_ports() if port in vals}
    return in_vals

def from_input_list(fwd, inv, input_batch, port_attr=None):
    """
    [input] -> [inputs, params, outputs].
    optionally if port_attr is already computed, pass it in to save time.
    """
    if port_attr is None:
        port_attr = propagate(inv)
    params = []
    outputs = []
    for input_data in input_batch:
        params_bwd = apply_backwards(inv, input_data, port_attr=port_attr)
        params_list = [params_bwd[port] for port in inv.in_ports() if is_param_port(port)]
        outputs_list = apply(fwd, input_data)
        params.append(params_list)
        outputs.append(outputs_list)
    return list(zip(input_batch, params, outputs))
