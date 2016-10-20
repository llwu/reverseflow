class Value:
    pass


class MutableValue(Value):
    def is_mutable() -> bool:
        return True

    def __init__(value) -> None:
        self.value = value


class ImmutableValue(Value):
    def is_mutable() -> bool:
        return False

    def __init__(value) -> None:
        self.type = 'float32'
        self.value = value
