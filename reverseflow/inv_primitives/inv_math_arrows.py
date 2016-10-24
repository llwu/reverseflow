from reverseflow.arrows.arrow import Arrow
from reverseflow.arrows.parametricarrow import ParametricArrow
from reverseflow.arrows.port import InPort, OutPort, ParamPort
from reverseflow.util.mapping import Bimap, ImageBimap, OneToMany, OneToManyList
from reverseflow.arrows.primitive.math_arrows import (AddArrow, SubArrow,
    LogArrow)
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow


class InvAddArrow(ParametricArrow):

    def __init__(self):
        name = "InvAdd"
        # consider having theta be something other than an InPort
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        sub = SubArrow()

        in_ports = sub.in_ports[0]
        out_ports = [sub.out_ports[0], dupl_theta.out_ports[1]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], sub.in_ports[1])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         param_ports=param_ports,
                         name=name)

InvAddArrowSet = OneToManyList()  # type: ImageBimap[Arrow, Tuple(Arrow, Dict[Int, Int])]
InvAddArrowSet.add(AddArrow, (InvAddArrow, {}))
InvAddArrowSet.add(AddArrow, (SubArrow, {0: 1}))
InvAddArrowSet.add(AddArrow, (SubArrow, {1: 1}))


class InvSubArrow(ParametricArrow):
    def __init__(self):
        name = "InvSub"
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        add = AddArrow()
        in_ports = add.in_ports[0]
        out_ports = [add.out_ports[0], dupl_theta.out_ports[1]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], add.in_ports[1])
        super().__init__(edges=edges, in_ports=in_ports,
                         out_ports=out_ports, param_ports=param_ports,
                         name=name)

InvSubArrowSet = OneToManyList()
InvSubArrowSet.add(SubArrow, (InvSubArrow, {}))
InvSubArrowSet.add(SubArrow, (AddArrow, {0: 0}))
InvSubArrowSet.add(AddArrow, (SubArrow, {1: 0}))

class InvMulArrow(ParametricArrow):

    def __init__(self):
        name = "InvMulAdd"
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        div = DivArrow()

        in_ports = div.in_ports[0]
        out_ports = [div.out_ports[0], dupl_theta.out_ports[1]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], div.in_ports[1])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         param_ports=param_ports,
                         name=name)

InvMulArrowSet = OneToManyList()  # type: ImageBimap[Arrow, Tuple(Arrow, Dict[Int, Int])]
InvMulArrowSet.add(MulArrow, (InvMulArrow, {}))
InvMulArrowSet.add(MulArrow, (SubArrow, {0: 1}))
InvMulArrowSet.add(MulArrow, (SubArrow, {1: 1}))



class InvExpArrow(ParametricArrow):
    def __init__(self) -> None:
        name = 'InvExp'
        # consider having theta be something other than an InPort
        edges = Bimap()  # type: EdgeMap
        dupl_theta = DuplArrow()
        log = LogArrow()

        in_ports = log.in_ports[1]
        out_ports = [dupl_theta.out_ports[1], log.out_ports[0]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], log.in_ports[0])
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         param_ports=param_ports,
                         name=name)

    @staticmethod
    def inverse_of() -> Arrow:
        """What is this the inverse of"""
        return ExpArrow

    def procedure(self):
        # consider having theta be something other than an InPort
        edges = Bimap() #type: EdgeMap
        dupl_theta = DuplArrow()
        log = LogArrow()

        in_ports = log.in_ports[1]
        out_ports = [dupl_theta.out_ports[1], log.out_ports[0]]
        param_ports = [dupl_theta.in_ports[0]]
        edges.add(dupl_theta.out_ports[0], log.in_ports[0])
        inv_exp = ParametricArrow(edges = edges,
                                  in_ports = in_ports,
                                  out_ports = out_ports,
                                  param_ports = param_ports)
        return inv_exp
