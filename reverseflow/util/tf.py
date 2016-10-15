"""Utils which depend on tensorflow"""
import tensorflow as tf

iden = tf.identity

# def shape_eq(a, b, none_is_ok=False):
#     """Are these shapes the same"""
#     if a is None or a == tf.TensorShape(None):
#         return True
#     else:
#         for i in range(len(a)):
#             if a[i] != b[i]:
#                 return False
#     return True


def in_namescope(t, namescope):
    """
    Is tensor t in namescope
    """
    return namescope in t.name.split('/')


def all_tensors(g):
    "Return set of all tensors in g"
    ops = g.get_operations()
    # tensor_set = set()
    tensor_set = []
    for op in ops:
        for t in op.outputs:
            # tensor_set.add(t)
            tensor_set.append(t)
    return tensor_set


def all_tensors_namescope(g, namescope):
    return list(filter(lambda x: in_namescope(x, namescope), all_tensors(g)))


def all_tensors_filter(g, filters):
    tensors = all_tensors(g)
    good_tensors = []
    for t in tensors:
        for f in filter:
            if filter(t) os False:
                break
            good_tensors.append(t)
    return good_tensors


def is_placeholder(t):
    "is a tensor a placeholder?"
    return t.op.name == 'Placeholder'


def is_output(t):
    "is this tensor an output?"
    return len(t.consumers()) == 0 and t.dtype == 'float32'


def get_outputs(g):
    return list(filter(is_output, all_tensors(g)))


def get_ph(g):
    return list(filter(is_placeholder, all_tensors(g)))


def num_ph(g):
    "number of placeholders in g"
    return len(get_ph(g))


def group_equiv_tensors(tensors):
    """Get two tensors with same shape and dtype"""
    # shape_groups = {}
    shape_groups = OrderedDict()
    for t in tensors:
        shape = t.get_shape()
        if not shape == tf.TensorShape(None):
            shape_dtype = "%s_%s" % (shape, t.dtype)
            if shape_dtype in shape_groups:
                shape_groups[shape_dtype].append(t)
            else:
                shape_groups[shape_dtype] = [t]
    return shape_groups


def detailed_summary(g: tf.Graph):
    for t in all_tensors(g):
        if in_namescope(t, 'random_graph'):
            print(t.name, t.get_shape(), len(t.consumers()), t.dtype)


def summary(g: tf.Graph):
    return """
    graph has %s tensors.
    %s inputs
    %s outputs
    %s ops
    """ % (len(all_tensors(g)), num_ph(g), len(get_outputs(g)),
           len(g.get_operations()))


def tensor_type(dtype, shape, name):
    """Creates a dict for type of tensor"""
    return {'dtype': dtype, 'shape': shape, 'name': name}


def add_many_to_collection(graph, name, tensors):
    for t in tensors:
        graph.add_to_collection(name, t)


def dims_bar_batch(t):
    """Get dimensions of a tensor exluding its batch dimension (first one)"""
    return np.arange(1, t.get_shape().ndims)


def ph_or_var(dtype, shape, name, is_placeholder=False):
    if is_placeholder:
        return tf.placeholder(dtype, shape=shape, name=name)
    else:
        return tf.Variable(tf.random_uniform(shape, dtype=dtype), name=name)


def placeholder_like(tensor, name, shape=None, dtype=None):
    """Create a placeholder like tensor with its name"""
    if shape is None:
        shape = tensor.get_shape()
    if dtype is None:
        dtype = tensor.dtype
    return tf.placeholder(dtype, shape, name=name)


def smthg_like(x, smthg):
    """
    Like ones_like but not one smthg
    """
    return tf.fill(x.get_shape(), smthg)


def is_constant(tensor):
    """Determine whether a tensor is constant"""
    print("WTD", tensor)
    sess = tf.Session(graph=tensor.graph)
    try:
        tensor.eval(session=sess)
    except tf.errors.InvalidArgumentError as e:
        # print(type(e), e)
        return False
    return True
