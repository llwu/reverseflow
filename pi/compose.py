import tensorflow as tf

def shape_eq(a, b, none_is_ok=False):
    """Are these shapes the same"""
    if a is None or a == tf.TensorShape(None):
        return True
    else:
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
    return True

not shape_eq((1, 2, 3), (1, 2, None))
shape_eq(TensorShape(1, 2), (1,2))
shape_eq(TensorShape(1, 2), (1,2), none_is_ok=True)
shape_eq(TensorShape(None, 2), TensorShape(1, 2), none_is_ok=True)


def compose(outputs, inputs, check_shape_matches):
    """
    Take 'outputs' and pipe them into 'inputs'.
    On a graph.
    Why do i need this.
    I need this because I need to pipe in the neural net to the parameters.
    Also sometimes i need to pipe the output into the input.
    Bit other times i need to just o the piping tmeporaril.

    So I should construct a new graph.
    """
    assert not shape_eq(check_shape_matches), "Shapes don't match"
    pass

    
