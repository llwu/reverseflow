from typing import Tuple, Dict
from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.port import InPort, OutPort, ParamPort
from reverseflow.util.mapping import Bimap, ImageBimap, OneToMany, OneToManyList
from reverseflow.arrows.primitive.math_arrows import (AddArrow, SubArrow,
    MulArrow, DivArrow, NegArrow)
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow


class InvAddArrow(CompositeArrow):
    """
    Parametric Inverse Addition
    add-1(z; theta) = (z-theta, theta)
    """

    def __init__(self):
        name = "InvAdd"
        # consider having theta be something other than an InPort
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        sub = SubArrow()

        in_ports = [sub.in_ports[0], dupl_theta.in_ports[0]]
        out_ports = [sub.out_ports[0], dupl_theta.out_ports[1]]
        edges.add(dupl_theta.out_ports[0], sub.in_ports[1])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         name=name)
        self.change_in_port_type(ParamPort, 1)


class InvSubArrow(CompositeArrow):
    """
    Parametric Inverse Subtraction
    sub-1(z; theta) = (z+theta, theta)
    """
    def __init__(self):
        name = "InvSub"
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        add = AddArrow()
        in_ports = [add.in_ports[0]]
        out_ports = [add.out_ports[0], dupl_theta.out_ports[1]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], add.in_ports[1])
        super().__init__(edges=edges, in_ports=in_ports,
                         out_ports=out_ports, param_ports=param_ports,
                         name=name)

class InvMulArrow(CompositeArrow):

    def __init__(self):
        name = "InvMulAdd"
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        div = DivArrow()

        in_ports = [div.in_ports[0]]
        out_ports = [div.out_ports[0], dupl_theta.out_ports[1]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], div.in_ports[1])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         param_ports=param_ports,
                         name=name)

class InvDivArrow(CompositeArrow):
    def __init__(self):
        name = "InvDivAdd"
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        div = DivArrow()

        in_ports = [div.in_ports[0]]
        out_ports = [div.out_ports[0], dupl_theta.out_ports[1]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], div.in_ports[1])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         param_ports=param_ports,
                         name=name)


class InvPowArrow(CompositeArrow):
    def __init__(self) -> None:
        name = 'InvPow'
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        log = LogArrow()

        in_ports = [log.in_ports[1]]
        out_ports = [dupl_theta.out_ports[1], log.out_ports[0]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], log.in_ports[0])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         param_ports=param_ports,
                         name=name)
