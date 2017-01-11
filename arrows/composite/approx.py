from arrows.composite.math import MeanArrow, VarFromMeanArrow
from arrows.compositearrow import CompositeArrow
from arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap


class ApproxIdentityArrow(CompositeArrow):
    """Approximate Identity Arrow
    f(x_1,..,x_n) = mean(x_1,,,,.x_n), var(x_1, ..., x_n)

    Last out_port is the error port
    """

    def __init__(self, n_inputs: int):
        name = "ApproxIdentity"
        edges = Bimap()  # type: EdgeMap
        mean = MeanArrow(n_inputs)
        varfrommean = VarFromMeanArrow(n_inputs)
        dupls = [DuplArrow() for i in range(n_inputs)]
        for i in range(n_inputs):
            edges.add(dupls[i].get_out_ports()[0], mean.get_in_ports()[i])
            edges.add(dupls[i].get_out_ports()[1], varfrommean.get_in_ports()[i+1])
        mean_dupl = DuplArrow(n_duplications=n_inputs+1)
        edges.add(mean.get_out_ports()[0], mean_dupl.get_in_ports()[0])
        edges.add(mean_dupl.get_out_ports()[n_inputs], varfrommean.get_in_ports()[0])
        out_ports = mean_dupl.get_out_ports()[0:n_inputs]
        error_ports = [varfrommean.get_out_ports()[0]]
        # x = varfrommean.get_out_ports()[0]
        # error_ports = [ErrorPort(x.arrow, x.index)]
        out_ports = out_ports + error_ports
        super().__init__(edges=edges,
                         in_ports=[dupl.get_in_ports()[0] for dupl in dupls],
                         out_ports=out_ports,
                         name=name)
        self.add_out_port_attribute(len(out_ports)-1, "Error")
        # self.change_out_port_type(ErrorPort, len(out_ports)-1)
