"""Arrow which can represent any TensorFlow graph."""
from arrows.primitivearrow import PrimitiveArrow


class TfArrow(PrimitiveArrow):
    """TensorFlow arrow. Can compile to any TensorFlow graph."""

    def is_tf(self) -> bool:
        return True

    def __init__(self, n_in_ports: int, n_out_ports: int) -> None:
        name = 'TfArrow'
        super().__init__(n_in_ports, n_out_ports, name=name)
