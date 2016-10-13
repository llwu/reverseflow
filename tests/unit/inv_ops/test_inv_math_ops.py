## Test the parametric inverses
import reverseflow.inv_ops.inv_math_ops as inv_math_ops

def approx_eq(a, b, tolerance=1e-5):
    """is a approximately equal to b"""
    return abs(a-b) < tolerance

def test_sound_pi():
    """Test a parametric inverse is sound using randomly sampled inputs"""
    tf.InteractiveSession()
    z = tf.placeholder(shape=(), dtype='float32')
    params = pi.param_gen(z)
    rand_params = gen(pinv.param_space)
    rand_inp = gen(pinv.inp_type)
    inv_output = pinv.inv_f(rand_inp..., rand_params...)
    fwd_output = pinv.inverse_of(inv_output...)
    fwd_output, rand_inp
    ≈(test_pinv(pi)...)

def verify_sound_pi():
    """Verify that a parametric inverse is sound"""
    pass

def verify_complete_pi():
    """Verify that a parametric inverse is complete"""
    pass
