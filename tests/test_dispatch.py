import numpy as np

from arrows.apply.propagate import propagate
from arrows.primitive.array_arrows import GatherArrow
from reverseflow.dispatch import inv_gather

def test_inv_gather():
    inds = {'shape': (3), 'value': np.array([0, 2, 4])}
    inp = {'shape': (5), 'value': np.array([0, 2, 1, 3, 9])}
    gather = GatherArrow()
    i = gather.get_in_ports()
    port_attrs = {i[0]: inp, i[1]: inds}
    arrow, portmap = inv_gather(gather, port_attrs)
    output = {'shape': (3), 'value': np.array([0, 1, 9])}
    theta = {'shape': (2), 'value': np.array([2, 3])}
    i = arrow.get_in_ports()
    inv_attrs = {i[0]: output, i[1]: theta}
    propd_values = propagate(arrow, inv_attrs)
    return inds, inp, port_attrs, arrow, portmap, propd_values

if __name__ == '__main__':
    inds, inp, port_attrs, arrow, portmap, propd_values = test_inv_gather()
    import pdb; pdb.set_trace()
