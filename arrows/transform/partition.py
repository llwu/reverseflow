"""Partition an arrow into layers of sub_arrows"""
from pqdict import pqdict
from arrows.arrow import Arrow
from arrows.compositearrow import CompositeArrow
from typing import List, Set

def partition(comp_arrow: CompositeArrow) -> List[Set[Arrow]]:
	partition_arrows = []
	arrow_colors = pqdict()
	for sub_arrow in comp_arrow.get_sub_arrows():
		arrow_colors[sub_arrow] = sub_arrow.num_in_ports()


	for port in comp_arrow.get_in_ports():
		in_ports = comp_arrow.neigh_in_ports(port)
		for in_port in in_ports:
			assert in_port.arrow in arrow_colors, "sub_arrow not in arrow_colors"
			arrow_colors[in_port.arrow] -= 1

	while len(arrow_colors) > 0:
		arrow_layer = set()
		view_arrow, view_priority = arrow_colors.topitem()

		while view_priority == 0:
			sub_arrow, priority = arrow_colors.popitem()
			arrow_layer.add(sub_arrow)
			if len(arrow_colors) == 0:
				break
			view_arrow, view_priority = arrow_colors.topitem()

		partition_arrows.append(arrow_layer)

		for arrow in arrow_layer:
			for out_port in arrow.get_out_ports():
				in_ports = comp_arrow.neigh_in_ports(out_port)
				for in_port in in_ports:
					if in_port.arrow != comp_arrow:
						assert in_port.arrow in arrow_colors, "sub_arrow not in arrow_colors"
						arrow_colors[in_port.arrow] -= 1

	return partition_arrows

