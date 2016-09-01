import tensorflow as tf

class Inverse():
    """Inverse Function"""
    def name(self):
        attribs = ["pinv"] if self.has_params else ["inj"]
        if self.is_approx: attribs.append("aprx")
        attribs.append(self.type)
        return "_".join(attribs)

class ParametricInverse(Inverse):
    """A parametric inverse"""
    def __init__(self, type, invf, param, is_approx):
        """
        type :: tf.Operation.type - type of forward operation
        param :: - generates params
        invf :: computes the inverse function
        is_approx ::
        """
        self.type = type
        self.param = param
        self.invf = invf

    def go(self, graph, inputs):
        print("PINV INP", inputs)
        # What about error ouputs
        with graph.as_default():
            with graph.name_scope(self.name()):
                params = self.param(inputs)
                ops = self.invf(inputs, params=params)
                return ops, params

class Injection(Inverse):
    """Invertible (Injective) Fucntion"""
    def __init__(self, type, invf):
        self.type = type
        self.invf = invf

    def go(self, graph, inputs, fwd_inputs, constants):
        print("INJ INP", inputs)
        # What about error ouputs
        with graph.as_default():
            with graph.name_scope("inj_%s" % self.type):
                ops, corres = self.invf(inputs, fwd_inputs, constants)
                return ops, corres

## Parameter Spaces
## ================
class ParameterSpace():
    """A parameter space"""
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
