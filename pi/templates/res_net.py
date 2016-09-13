import tensorflow as tf
import pi.util
import numpy as np

def consistent_batch_size(shapes):
    """Are the batch sizes the same"""
    return pi.util.same([shape[0] for shape in shapes])

def batch_flatten(tensors):
    """Flatten a vector of tensors, checking batch sizes are same"""
    return [tf.reshape(t,[t.get_shape()[0].value,-1]) for t in tensors]


def res_net_template_dict(inputs, out_shapes, **kwargs):
    input_list = list(inputs.values())
    out_shapes_list = list(out_shapes.values())
    outputs, params = res_net_template(input_list, out_shapes_list, **kwargs)
    return dict(zip(out_shapes.keys(), outputs)), params

def res_net_template(inputs, out_shapes, **kwargs):
    """
    Residual neural network inputs
    inputs : [tf.Tensor/tf.Variable] - inputs to by transformed
    out_shapes : (tf.TensorShape) | (Int) - shapes of output of tensor (includes batch_size)
    """
    # layer_width = kwargs['layer_width']
    # nblocks = kwargs['nblocks']
    # block_size = kwargs['block_size']
    # output_args = kwargs['output_args']
    inp_shapes = [x.get_shape().as_list() for x in inputs]
    assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"
    flat_input_shapes = [np.prod(inp_shape[1:]) for inp_shape in inp_shapes]
    input_width = np.sum(flat_input_shapes)

    flat_output_shapes = [np.prod(out_shape[1:]) for out_shape in out_shapes]
    output_width = np.sum(flat_output_shapes)

    # Num inputs and outputs
    ninputs = len(inp_shapes)
    noutputs = len(out_shapes)

    params = []

    flat_inputs = batch_flatten(inputs)
    x = tf.concat(1, flat_inputs)

    ## Layers
    W = tf.Variable(tf.zeros([input_width, output_width]), name="W")
    b = tf.Variable(tf.zeros([output_width]), name="b")
    l1_op = tf.matmul(x, W) + b
    params = params + [W, b]

    output_product = l1_op
    ## Unconcatenate output and separate
    outputs = []
    lb = 0
    for i in range(noutputs):
        ub = lb + flat_output_shapes[i]
        out = output_product[:, lb:ub]
        new_shape = [out.get_shape().as_list()[0]] + out_shapes[i][1:]
        print("newshape", new_shape)
        rout = tf.reshape(out, new_shape)
        outputs.append(rout)
        lb = ub

    return outputs, params

def test_res_net():
    batch_size = 128
    x = tf.placeholder(tf.float32, shape=(batch_size, 10))
    y = tf.placeholder(tf.float32, shape=(batch_size, 10, 20))
    inputs = (x,y)
    output_shapes = [(batch_size, 20), (batch_size, 30)]
    outputs, params = res_net_template(inputs, output_shapes)

    ## Run the damn thing
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    feed = {t:np.random.rand(*t.get_shape().as_list()) for t in inputs}
    sess.run(outputs, feed_dict=feed)

# test_res_net()
