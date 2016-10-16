from reverseflow.arrows.arrow import Arrow


class ParametricArrow(Arrow):
    """Parametric arrow"""

    def is_parametric() -> bool:
        return True

    def __init__(self):
        self.param_inport = []  # type: List[ParamInPort]
        # self.error_outport = []  # type: List[ErrorOutPort]
