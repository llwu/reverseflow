"""Arrow which can represent any TensorFlow graph."""

import tensorflow as tf

from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.port import InPort, OutPort


class TfArrow(Arrow):
    """TensorFlow arrow. Can compile to any TensorFlow graph."""

    def is_tf(self) -> bool:
        return True

    def __init__(self, graph: tf.Graph, name: str) -> None:
        super().__init__(name=name)
        self.n_in_ports = 1
        self.n_out_ports = 1
        self.in_ports = [InPort(self, 0)]
        self.out_ports = [OutPort(self, 0)]
        self.graph = graph

    def redefine_as(self, graph: tf.Graph) -> None:
        self.graph = graph
