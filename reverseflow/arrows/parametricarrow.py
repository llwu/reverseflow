from typing import List
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.port import InPort, OutPort, ParamPort


class ParametricArrow(CompositeArrow):
    """Parametric arrow"""

    def is_parametric(self) -> bool:
        return True

    def __init__(self,
                 edges: EdgeMap,
                 in_ports: List[InPort],
                 out_ports: List[OutPort],
                 param_ports: List[ParamPort]):
        super().__init__(edges=edges, in_ports=in_ports, out_ports=out_ports)
        self.param_ports = param_ports  # type: List[ParamPort]
