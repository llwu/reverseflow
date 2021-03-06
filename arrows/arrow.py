"""Major classes for Arrow data structures"""

import arrows.port_attributes as pa
import arrows.apply.constants as co
import arrows.apply.shapes as shp

class Arrow:
    """Abstract arrow class"""

    def __init__(self, name: str, parent=None) -> None:
        self.name = name
        self.parent = parent

    def get_port_attr(self, port):
        assert port.arrow is self
        return self.port_attr[port.index]

    def ports(self, idx=None):
        if idx is None:
            return self._ports
        else:
            return [self._ports[i] for i in idx]

    def all_ports(self):
        allports = list(self._ports)
        for arrow in self.get_sub_arrows():
            allports += arrow.all_ports()
        return allports

    def port(self, index: int):
        return self.ports()[index]

    def in_ports(self, idx=None):
        """
        Get InPorts of an Arrow.
        Returns:
            List of InPorts
        """
        return [port for port in self._ports if pa.is_in_port(port)]

    def in_port(self, index: int):
        """
        Get ith InPort
        """
        return self.in_ports()[index]

    def param_ports(self):
        """
        Get ParamPorts of an Arrow.
        Returns:
            List of ParamPorts
        """
        return [port for port in self._ports if pa.is_param_port(port)]

    def error_ports(self):
        """
        Get ErrorPorts of an Arrow.
        Returns:
            List of ErrorPorts
        """
        return [port for port in self._ports if pa.is_error_port(port)]

    def out_ports(self):
        """
        Get OutPorts of an Arrow.
        Returns:
            List of OutPorts
        """
        return [port for port in self._ports if pa.is_out_port(port)]

    def out_port(self, index: int):
        """
        Get ith OutPort
        """
        return self.out_ports()[index]

    def num_ports(self) -> int:
        return len(self.ports())

    def num_in_ports(self) -> int:
        return len(self.in_ports())

    def num_out_ports(self) -> int:
        return len(self.out_ports())

    def num_param_ports(self) -> int:
        return len(self.param_ports())

    def num_error_ports(self) -> int:
        return len(self.error_ports())

    def is_primitive(self) -> bool:
        return False

    def is_composite(self) -> bool:
        return False

    def is_source(self) -> bool:
        return False

    def is_parametric(self) -> bool:
        return False

    def is_approximate(self) -> bool:
        return False

    def is_tf(self) -> bool:
        return False

    def get_sub_arrows(self):
        return []

    def __deepcopy__(self, memo):
        return None

    def get_dispatches(self):
        return {co.constant_pred: co.constant_dispatch,
                shp.val_to_shape_pred: shp.val_to_shape_dispatch}

    def get_topo_order(self):
        if self.parent is None:
            return [self.topo_order]
        else:
            return self.parent.get_topo_order() + [self.topo_order]
