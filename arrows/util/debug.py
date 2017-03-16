from arrows.port_attributes import is_in_port

def down(port, i=0):
    if is_in_port(port):
        neighs = port.arrow.parent.neigh_ports(port)
        if len(neighs) > i:
            return neighs[i]
        else:
            return None
    else:
        ins = port.arrow.in_ports()
        if len(ins) > i:
            return ins[i]
        else:
            return None

def up(port, i=0):
    if is_in_port(port):
        outs = port.arrow.out_ports()
        if len(outs) > i:
            return outs[i]
        else:
            return None
    else:
        neighs = port.arrow.parent.neigh_ports(port)
        if len(neighs) > i:
            return neighs[i]
        else:
            return None

def side(port, i=0):
    if is_in_port(port):
        ins = port.arrow.in_ports()
        if len(ins) > i:
            return ins[i]
        else:
            return None
    else:
        outs = port.arrow.out_ports()
        if len(outs) > i:
            return outs[i]
        else:
            return None

def find(comp_arrow, name):
    return [a for a in comp_arrow.get_sub_arrows() if name in str(a)]
