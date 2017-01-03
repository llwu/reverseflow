from numpy import ndarray
from typing import List
import tensorflow as tf
from arrows.arrow import Arrow
from reverseflow.to_graph import arrow_to_graph
from arrows.config import floatX

def apply(arrow: Arrow, inputs: List[ndarray]) -> List[ndarray]:
    """Apply an arrow to some inputs"""
    assert len(inputs) == arrow.num_in_ports(), "wrong # inputs"
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with graph.as_default():
        input_tensors = [tf.placeholder(dtype=floatX()) for i in range(len(inputs))]
        outputs = arrow_to_graph(arrow, input_tensors)
        feed_dict = dict(zip(input_tensors, inputs))
        init = tf.initialize_all_variables()
        sess.run(init)
        outputs = sess.run(fetches=outputs,
                           feed_dict=feed_dict)
    sess.close()
    return outputs
