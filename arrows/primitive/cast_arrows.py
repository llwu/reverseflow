from typing import Dict

from arrows.apply.shapes import shape_pred, shape_dispatch
from arrows.primitivearrow import PrimitiveArrow

class CastArrow(PrimitiveArrow):
    def __init__(self, to_dtype):
        name = 'Cast'
        self.to_dtype = to_dtype
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)

    def get_dispatches(self):
        disp = super().get_dispatches()
        disp.update({shape_pred: shape_dispatch})
        return disp
