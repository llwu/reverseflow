"""
Partition an arrow into layers of sub_arrows
Attach function approximators (NN) to each layer
"""
from arrows.arrow import Arrow
from arrows.tfarrow import TfArrow
from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import is_in_port, is_param_port, is_out_port
from typing import List, Set

from pqdict import pqdict
from copy import deepcopy
import tensorflow as tf


def partition(comp_arrow: CompositeArrow) -> List[Set[Arrow]]:
	"""Partitions the comp_arrow into sequential layers of its sub_arrows"""
	partition_arrows = []
	arrow_colors = pqdict()
	for sub_arrow in comp_arrow.get_sub_arrows():
		arrow_colors[sub_arrow] = sub_arrow.num_in_ports()


	for port in comp_arrow.in_ports():
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
			for out_port in arrow.out_ports():
				in_ports = comp_arrow.neigh_in_ports(out_port)
				for in_port in in_ports:
					if in_port.arrow != comp_arrow:
						assert in_port.arrow in arrow_colors, "sub_arrow not in arrow_colors"
						arrow_colors[in_port.arrow] -= 1

	return partition_arrows


def attachNN(comp_arrow: CompositeArrow) -> CompositeArrow:
	"""
	Returns a composite arrow with the neural networks already
	attached to each layer of sub_arrows
	"""
	new_arrow = deepcopy(comp_arrow)
	partition_arrows = partition(new_arrow)

	for (i, layer) in enumerate(partition_arrows):
		in_ports = []
		param_ports = []

		# input ports of new_arrow are inputs of the first neural network to
		# provide information to the sub_arrows with only parametric inputs
		if i == 0:
			for in_port in new_arrow.in_ports():
				if is_param_port(in_port) == False:
					in_ports.extend(new_arrow.neigh_in_ports(in_port))

		for sub_arrow in layer:
			for port in sub_arrow.in_ports():
				for neigh_port in new_arrow.neigh_out_ports(port):
					if is_param_port(neigh_port):
						if port not in param_ports:
							param_ports.append(port)
					elif neigh_port.arrow == new_arrow and is_in_port(neigh_port):
						if port not in in_ports:
							in_ports.append(port)
					elif neigh_port.arrow != new_arrow and is_out_port(neigh_port):
						if port not in in_ports:
							in_ports.append(port)


		if len(in_ports) == 0 or len(param_ports) == 0:
			continue

		neural_net_arrow = TfArrow(n_in_ports=len(in_ports),
									n_out_ports=len(param_ports),
									graph=tf.Graph(),
									name="nn_for_params_"+str(i))

		nn_in_ports = neural_net_arrow.in_ports()
		nn_out_ports = neural_net_arrow.out_ports()

		for (j, in_port) in enumerate(in_ports):
			new_arrow.add_edge(in_port, nn_in_ports[j])
		for (j, param_port) in enumerate(param_ports):
			new_arrow.add_edge(nn_out_ports[j], param_port)

	return new_arrow
