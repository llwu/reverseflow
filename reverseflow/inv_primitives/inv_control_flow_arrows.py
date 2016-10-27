from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.parametricarrow import ParametricArrow
from reverseflow.arrows.port import InPort, OutPort, ParamPort
from reverseflow.util.mapping import Bimap
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow

class InvDuplArrow(ParametricArrow):
    def __init__(self):
        self.name = "InvDupl"

    @staticmethod
    def inverse_of() -> Arrow:
        """What is this the inverse of"""
        return DuplArrow

    def procedure(self):
        # consider having theta be something other than an InPort
        edges = Bimap()  # type: EdgeMap
        identity = IdentityArrow()

        in_ports = [identity.in_ports[0]]
        out_ports = [identity.out_ports[0]]
        param_ports = []
        inv_dupl = ParametricArrow(edges=edges,
                                  in_ports=in_ports,
                                  out_ports=out_ports,
                                  param_ports=param_ports)
        return inv_dupl
