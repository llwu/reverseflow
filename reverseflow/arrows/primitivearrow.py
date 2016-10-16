from reverseflow.arrows.arrow import Arrow


class PrimitiveArrow(Arrow):
    """Primitive arrow"""
    def is_primitive(self) -> bool:
        return True

    def __init__(self) -> None:
        pass
