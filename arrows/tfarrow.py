"""Arrow which can represent any TensorFlow graph."""
from arrows.primitivearrow import PrimitiveArrow
from tensortemplates.res_net import template as res_net_template

class TfArrow(PrimitiveArrow):
    """TensorFlow arrow. Can compile to any TensorFlow graph."""

    def is_tf(self) -> bool:
        return True

    def __init__(self, n_in_ports: int, n_out_ports: int,
                 template=None, options=None) -> None:
        name = 'TfArrow'
        self.template = res_net_template if template is None else template
        if options:
            self.options = options
        else:
            self.options = {'layer_width': 10,
                            'nblocks': 1,
                            'block_size': 2,
                            'reuse': False}
        super().__init__(n_in_ports, n_out_ports, name=name)


class TfLambdaArrow(PrimitiveArrow):
    """TensorFlow arrow. Can compile to any TensorFlow graph."""

    def is_tf(self) -> bool:
        return True

    def __init__(self, n_in_ports: int, n_out_ports: int, func) -> None:
        name = 'TfArrow'
        self.func = func
        super().__init__(n_in_ports, n_out_ports, name=name)
