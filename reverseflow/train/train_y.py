from reverseflow.arrows.parametricarrow import ParametricArrow
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph

from typing import List
from tensorflow import Graph, Tensor


def train_y_tf(outputs: List[Tensor]) -> Graph:
    """
    """
    arrow = graph_to_arrow(outputs)
    inv_arrow = invert(arrow)
    train_y(inv_arrow)


def train_y_arr(arrow: Arrow, dataset: List):
    inv_arrow = invert(arrow)
    train_y_parr(inv_arrow, dataset)
    # if necessary append tensorflow arrow

def reduce_approx_error(approx_arrow: Arrow) -> Arrow:
    """
    From approximate arrow with n error symbols of arbitrary shape
    reduce to single scalar error
    """

def train_y_parr(p_arrow: ParametricArrow, dataset: List) -> ParametricArrow:
    """
    Given parametric arrow p_arrow : X -> Y
    and dataset Y
    """
    reduced_arrow, cost_out_port = reduce_approx_error(p_arrow)
    graph, input_tensors, output_tensors = to_graph(tensorflow)
    loss_tensors = ...
    loss_tensor = ...some reduction

    optimizer = tf.train.MomentumOptimizer(learning_rate=options['learning_rate'],
                                               momentum=options['momentum'])
    update_step = optimizer.minimize(loss)
    train_loop(num_iterations)
    # add loss function
    convert to tensorflow graph
    # train
    # convert back to arrow

def train_loop(num_iterations:int):
    for i in range(num_iterations):
        ...
