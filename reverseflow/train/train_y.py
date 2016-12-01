from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.to_graph import arrow_to_graph

from typing import List
from tensorflow import Graph, Tensor

# When to add c function in inversion or otherwise
# - Ideally not at all, then when all other methods fail then add
# For now assume its added

# How to get tensors corresponding to error out_ports
# - Update to_graph to expect both parametric inputs and error outputs
# should return parameter_tensors and error output tensors



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

def train_y_arr(param_arrow: CompositeArrow, dataset: List) -> ParametricArrow:
    """
    Given:

    param_arrow : Y x Theta -> X
    dataset : [Y]

    Find Theta such that

    """
    # Convert to tensorflow
    # Find which tensors correspond to costs
    # in tensorflow reduce and minimize these to a single scalar
    # Need to find correspondance of parameters to parametric inputs
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
