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


def train_y_parr(p_arrow: ParametricArrow, dataset: List) -> ParametricArrow:
    """
    Given parametric arrow p_arrow : X -> Y
    and dataset Y
    """
    # add loss function
    convert to tensorflow graph
    # train
    # convert back to arrow
