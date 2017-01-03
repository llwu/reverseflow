"""Tests port equality and hashing."""

from arrows.port import InPort, OutPort, ParamPort, ErrorPort
from arrows.primitive.math_arrows import MulArrow, AddArrow

# This now needs to reflect fact that Ports of same type are equal even if different specialization
# i.e. ErrorPort(a,0) == OutPort(a,0)


# def port_test():
#     """Tests that ports are distinguished by arrow, index, and type."""
#     mul = MulArrow()
#     add = AddArrow()
#     port_list = []
#     port_set = set()
#     for _ in range(3):
#         for i in range(10):
#             port_list.append(InPort(mul, i))
#             port_list.append(OutPort(mul, i))
#             port_list.append(ParamPort(mul, i))
#             port_list.append(ErrorPort(mul, i))
#             port_list.append(InPort(add, i))
#             port_list.append(OutPort(add, i))
#             port_list.append(ParamPort(add, i))
#             port_list.append(ErrorPort(add, i))
#             port_set.add(InPort(mul, i))
#             port_set.add(OutPort(mul, i))
#             port_set.add(ParamPort(mul, i))
#             port_set.add(ErrorPort(mul, i))
#             port_set.add(InPort(add, i))
#             port_set.add(OutPort(add, i))
#             port_set.add(ParamPort(add, i))
#             port_set.add(ErrorPort(add, i))
#     assert len(port_list) == 240, 'port_list wrong length'
#     assert len(port_set) == 80, 'port_set wrong size'
#     for i in range(len(port_list)):
#         for j in range(len(port_list)):
#             assert (j - i) % 80 == 0 or port_list[i] != port_list[j], 'equality test failed'
#
# port_test()
