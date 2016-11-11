"""Arrow which can represent any TensorFlow graph."""

import tensorflow as tf

from reverseflow.arrows.arrow import PrimitiveArrow


class TfArrow(PrimitiveArrow):
    """TensorFlow arrow. Can compile to any TensorFlow graph."""

    def is_tf(self) -> bool:
        return True

    def __init__(self, n_in_ports: int, n_out_ports: int, graph: tf.Graph, name: str) -> None:
        super().__init__(n_in_ports, n_out_ports, name)
        self.graph = graph

    def redefine_as(self, graph: tf.Graph) -> None:
        self.graph = graph
