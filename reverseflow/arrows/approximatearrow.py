from typing import List
from reverseflow.arrows.compositearrow import CompositeArrow, EdgeMap
from reverseflow.arrows.port import InPort, OutPort, ErrorPort


class ApproximateArrow(CompositeArrow):
    """Approximate arrow
    Has an addition error output
    """

    def is_approximate() -> bool:
        return True

    def __init__(self,
                 edges: EdgeMap,
                 in_ports: List[InPort],
                 out_ports: List[OutPort],
                 error_ports: List[ErrorPort]):
        super().__init__(edges=edges, in_ports=in_ports, out_ports=out_ports)
        self.error_ports = error_ports  # type: List[ErrorPort]
