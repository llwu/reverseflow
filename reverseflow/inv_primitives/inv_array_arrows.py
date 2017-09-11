"""Constructors for inverse arrows."""

import tensorflow as tf

from arrows.compositearrow import CompositeArrow
from arrows.port_attributes import *
from arrows.primitive.control_flow import IgnoreInputArrow


def gathernd_bwd_pred(arr: "InvGatherNdArrow", port_attr: PortAttributes):
    return port_has(arr.in_port(2), 'value', port_attr) and port_has(arr.out_port(0), 'value', port_attr)


def gathernd_bwd_disp(arr: "InvGatherNdArrow", port_attr: PortAttributes):
    inds = tf.constant(port_attr[arr.in_port(2)]['value'])
    output = tf.constant(port_attr[arr.out_port(0)]['value'])
    gather_nd = tf.gather_nd(output, inds)
    with tf.Session() as sess:
        return {arr.in_ports()[0]: {'value': sess.run(gather_nd)}}



class InvGatherNdArrow(CompositeArrow):
    """
    Atm just returns theta and ignores input, so needs elim to work.
    TODO: change this maybe
    """

    def __init__(self):
        super().__init__(name="InvGatherNd")
        for _ in range(3):
            in_port = self.add_port()
            make_in_port(in_port)
        make_param_port(self.in_port(1))
        out_port = self.add_port()
        make_out_port(out_port)

        ii1 = IgnoreInputArrow()
        ii2 = IgnoreInputArrow()

        # there should maybe be an error term here if doing it this way
        # (though current elim is exact)
        self.add_edge(self.in_port(0), ii1.in_port(0))
        self.add_edge(self.in_port(1), ii1.in_port(1))
        self.add_edge(self.in_port(2), ii2.in_port(0))
        self.add_edge(ii1.out_port(0), ii2.in_port(1))
        self.add_edge(ii2.out_port(0), out_port)
        assert self.is_wired_correctly()

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({
            gathernd_bwd_pred: gathernd_bwd_disp
            })
        return disp
