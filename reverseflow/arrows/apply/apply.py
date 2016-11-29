from numpy import ndarray
from typing import List
import tensorflow as tf
from reverseflow.arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_new_graph

def apply(arrow: Arrow, inputs: List[ndarray], params: List[ndarray] = []) -> List[ndarray]:
    """Apply an arrow to some inputs"""
    assert len(inputs) == arrow.n_in_ports, "wrong # inputs"
    assert len(params) == arrow.n_param_ports, "wrong # param inputs"
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with graph.as_default():
        input_tensors = [tf.placeholder(dtype='float32') for i in range(len(inputs))]
        param_tensors = [tf.placeholder(dtype='float32') for i in range(len(params))]

    graph_etc = arrow_to_new_graph(arrow, input_tensors, param_tensors, graph)
    feed_dict = dict(zip(input_tensors + param_tensors, inputs + params))

    outputs = sess.run(fetches=graph_etc['output_tensors'],
                       feed_dict=feed_dict)
    sess.close()
    return outputs
