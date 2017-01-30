from typing import Dict

from arrows.primitivearrow import PrimitiveArrow

class CastArrow(PrimitiveArrow):
    def __init__(self, to_dtype):
        name='cast'
        self.to_dtype = to_dtype
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        # TODO: do we need to actually do something?
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = ptv[i[0]]
        if o[0] in ptv:
            ptv[i[0]] = ptv[o[0]]
        return ptv
