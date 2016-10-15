"""Decode an arrow into a tensoflow graph"""
import tensorflow as tf
from pqdict import pqdict


def arrow_to_graph(arrow: reverseflow.arrows.Arrow) -> tf.Graph:
    """Convert an arrow to a tensorflow graph"""
    graph = tf.Graph()

    # A priority queue for each sub_arrow
    # priority is the number of inputs it has which have already been 'seen'
    # 'seen' inputs are inputs to the composition, or outputs of arrows that
    # have already been converted into tensorfow
    arrow_colors = pqdict()
    for sub_arrow in arrow.get_arrows():
        arrow_colors[sub_arrow] = sub_arrow.num_inputs()

    # create a tensor for each inport to the composition
    # decrement priority for each arrow connected to inputs
    for sub_arrow in arrow.get_inports():
        tensor = tf.placeholder()
        arrow_colors[sub_arrow]  # TODO: decrement

    while len(arrows_colors) > 0:
        sub_arrow, priority = arrow_colors.pop()
        assert priority == 0, "All inputs to subarrow should be resolved first"
        assert sub_arrow.is_primitive(), "Convert unflat arrows unimplmented"
        op = graph.create_op(op_type=sub_arrow,
                             inputs=,
                             dtypes=,
                             name=)
    # pop arrow with zero
    # create correspoding tensorflow op
    # update arrow queue
