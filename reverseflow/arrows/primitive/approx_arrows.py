from reverseflow.arrows.compositearrow import CompositeArrow
from reverseflow.arrows.primitive.control_flow_arrows import DuplArrow
from reverseflow.util.mapping import Bimap


class ApproxIdentityArrow(CompositeArrow):
    """Approximate Identity Arrow
    f(x_1,..,x_n) = mean(x_1,,,,.x_n), var(x_1, ..., x_n)
    """

    def __init__(self, n_inputs: int):
        name = "ApproxIdentity"
        edges = Bimap()  # type: EdgeMap
        mean = MeanArrow()
        var = VarArrow()  # TODO Add this arrow
        dupls = [DuplArrow() for i in range(n_inputs)]

        # Since mean and variance share alot of sub computation, its
        # more efficient to not just use mean and var arrows independently
        for i in range(n_inputs):
            edges.add(dupls[i].out_ports[0], mean.in_ports[i])
            edges.add(dupls[i].out_ports[1], var.in_ports[i])
        mean_dupl = DuplArrow(n_duplications=n_inputs)
        edges.add(mean.out_ports[0], mean_dupl.in_ports[0])
        in_ports = [dupl.in_port[0] for dupl in dupls]
        out_ports = mean_dupl.out_ports
        error_ports = var.out_ports
        super().__init__(edges=edges,
                         in_ports=in_ports,
                         out_ports=out_ports,
                         error_ports=error_ports,
                         name=name)


def inv_split_approx(z):
    """Outputs the mean of its inputs and the error as the variance"""
    mean = tf.add_n(z)/len(z)
    variances = [tf.abs(t - mean) for t in z]
    mean_variances = tf.add_n(variances)/len(z)
    batched_error = tf.reduce_mean(mean_variances,
                                   reduction_indices=dims_bar_batch(mean_variances))
    return (mean,), (batched_error,)
