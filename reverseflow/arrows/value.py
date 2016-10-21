class Value:
    pass


class MutableValue(Value):
    def is_mutable(self) -> bool:
        return True

    def __init__(self, value) -> None:
        self.value = value


class ImmutableValue(Value):
    def is_mutable(self) -> bool:
        return False

    def __init__(self, value) -> None:
        self.value = value

    # FIXME: make immutable
