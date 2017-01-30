from typing import Dict

import numpy as np

from arrows.primitivearrow import PrimitiveArrow


class RangeArrow(PrimitiveArrow):
    """range"""

    def __init__(self):
        name = 'range [a,b]'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv and i[1] in ptv:
            ptv[o[0]] = np.arange(ptv[i[0]], ptv[i[1]])
        if o[0] in ptv and len(ptv[o[0]]) > 0:
            ptv[i[0]] = ptv[o[0]][0]
            ptv[i[1]] = ptv[o[0]][-1] + 1
        return ptv


class RankArrow(PrimitiveArrow):
    """Number of dimensions of an arrow"""

    def __init__(self):
        name = 'rank'
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def eval(self, ptv: Dict):
        i = self.get_in_ports()
        o = self.get_out_ports()
        if i[0] in ptv:
            ptv[o[0]] = np.linalg.matrix_rank(ptv[i[0]])
        return ptv
