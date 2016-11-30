from reverseflow.arrows.primitivearrow import PrimitiveArrow

class CastArrow(PrimitiveArrow):

    def __init__(self, to_dtype):
        name='cast'
        self.to_dtype = to_dtype
        super().__init__(n_in_ports=1, n_out_ports=1, name=name)
