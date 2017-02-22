""" (Inverse) Rendering"""
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from arrows.util.viz import show_tensorboard_graph, show_tensorboard
from arrows.util.misc import getn
from arrows.config import floatX
from benchmarks.common import handle_options, gen_sfx_key
from reverseflow.train.common import get_tf_num_params
from reverseflow.train.loss import inv_fwd_loss_arrow, supervised_loss_arrow
from reverseflow.train.unparam import unparam
import sys
import getopt
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1,
    competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be
    auto-generated.
    """
    # from realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return np.array(M, dtype=floatX())


# n random matrices
def rand_rotation_matrices(n):
    return np.stack([rand_rotation_matrix() for i in range(n)])

# Genereate values in raster space, x[i,j] = [i,j]
def gen_fragcoords(width, height):
    """Create a (width * height * 2) matrix, where element i,j is [i,j]
       This is used to generate ray directions based on an increment"""
    raster_space = np.zeros([width, height, 2], dtype=floatX())
    for i in range(width):
        for j in range(height):
            raster_space[i,j] = np.array([i,j], dtype=floatX()) + 0.5
    return raster_space

# Append an image filled with scalars to the back of an image.
def stack(intensor, width, height, scalar):
    scalars = np.ones([width, height, 1], dtype=floatX()) * scalar
    return np.concatenate([intensor, scalars], axis=2)


def switch(cond, a, b):
    return cond*a + (1-cond)*b


def dot(a, b):
    """Dot product of two a and b"""
    print("A", a)
    print("B", b)
    c = tf.reduce_sum(a * b)
    print("C", c.get_shape())
    return c


def norm(x):
    return np.linalg.norm(x, 2, axis=3)


def make_ro(r, raster_space, width, height):
    """Symbolically render rays starting with raster_space according to geometry
      e  defined by """
    nmatrices = r.shape[0]
    resolution = np.array([width, height], dtype=floatX())
    # Normalise it to be bound between 0 1
    norm_raster_space = raster_space / resolution
    # Put it in NDC space, -1, 1
    screen_space = -1.0 + 2.0 * norm_raster_space
    # Make pixels square by mul by aspect ratio
    aspect_ratio = resolution[0]/resolution[1]
    ndc_space = screen_space * np.array([aspect_ratio, 1.0], dtype=floatX())
    # Ray Direction

    # Position on z-plane
    ndc_xyz = stack(ndc_space, width, height, 1.0)*0.5  # Change focal length

    # Put the origin farther along z-axis
    ro = np.array([0, 0, 1.5], dtype=floatX())

    # Rotate both by same rotation matrix
    ro_t = np.dot(np.reshape(ro, (1, 3)), r)
    ndc_t = np.dot(np.reshape(ndc_xyz, (1, width, height, 3)), r)
    print(ndc_t.shape, width, height, nmatrices)
    ndc_t = np.reshape(ndc_t, (width, height, nmatrices, 3))
    ndc_t = np.transpose(ndc_t, (2, 0, 1, 3))

    # Increment by 0.5 since voxels are in [0, 1]
    ro_t = ro_t + 0.5
    ndc_t = ndc_t + 0.5
    # Find normalise ray dirs from origin to image plane
    unnorm_rd = ndc_t - np.reshape(ro_t, (nmatrices, 1, 1, 3))
    rd = unnorm_rd / np.reshape(norm(unnorm_rd), (nmatrices, width, height, 1))
    return rd, ro_t


def gen_img(voxels, rotation_matrix, width, height, nsteps, res):
    """Renders n voxel grids in m different views
    voxels : (n, res, res, res)
    rotation_matrix : (m, 4)
    returns (n, m, width, height))
    """
    raster_space = gen_fragcoords(width, height)
    rd, ro = make_ro(rotation_matrix, raster_space, width, height)
    a = 0 - ro  # c = 0
    b = 1 - ro  # c = 1
    nmatrices = rotation_matrix.shape[0]
    tn = np.reshape(a, (nmatrices, 1, 1, 3))/rd
    tff = np.reshape(b, (nmatrices, 1, 1, 3))/rd
    tn_true = np.minimum(tn, tff)
    tff_true = np.maximum(tn, tff)
    # do X
    tn_x = tn_true[:, :, :, 0]
    tff_x = tff_true[:, :, :, 0]
    tmin = 0.0
    tmax = 10.0
    t0 = tmin
    t1 = tmax
    t02 = np.where(tn_x > t0, tn_x, t0)
    t12 = np.where(tff_x < t1, tff_x, t1)
    # y
    tn_x = tn_true[:, :, :, 1]
    tff_x = tff_true[:, :, :, 1]
    t03 = np.where(tn_x > t02, tn_x, t02)
    t13 = np.where(tff_x < t12, tff_x, t12)
    # z
    tn_x = tn_true[:, :, :, 2]
    tff_x = tff_true[:, :, :, 2]
    t04 = np.where(tn_x > t03, tn_x, t03)
    t14 = np.where(tff_x < t13, tff_x, t13)

    # Shift a little bit to avoid numerial inaccuracies
    t04 = t04*1.001
    t14 = t14*0.999

    batched = len(voxels.get_shape()) > 1
    batch_size = int(voxels.get_shape()[0]) if batched else 1
    left_over = np.ones((batch_size, nmatrices * width * height,))
    step_size = (t14 - t04)/nsteps
    orig = np.reshape(ro, (nmatrices, 1, 1, 3)) + rd * np.reshape(t04,(nmatrices, width, height, 1))
    xres = yres = zres = res

    orig = np.reshape(orig, (nmatrices * width * height, 3))
    rd = np.reshape(rd, (nmatrices * width * height, 3))
    step_sz = np.reshape(step_size, (nmatrices * width * height, 1))
    print(voxels)
    # voxels = tf.reshape(voxels, [-1])

    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_sz*i
        voxel_indices = np.floor(pos*res)
        pruned = np.clip(voxel_indices,0,res-1)
        p_int = pruned.astype('int64')
        indices = np.reshape(p_int, (nmatrices*width*height,3))
        flat_indices = indices[:, 0] + res * (indices[:, 1] + res * indices[:, 2])
        # print("ishape", flat_indices.shape, "vshape", voxels.get_shape())
        # attenuation = voxels[:, indices[:,0],indices[:,1],indices[:,2]]
        attenuation = None
        if batched:
            x = np.arange(batch_size)
            batched_indices = np.transpose([np.repeat(x, len(flat_indices)),
                    np.tile(flat_indices, len(x))]).reshape(batch_size, len(flat_indices), 2)
            attenuation = tf.gather_nd(voxels, batched_indices)
        else:
            attenuation = tf.gather(voxels, flat_indices)
        print("attenuation step", attenuation.get_shape(), step_sz.shape)
        left_over = left_over*tf.exp(-attenuation*step_sz.reshape(nmatrices * width * height))

    img = left_over
    img_shape = tf.TensorShape((batch_size, nmatrices, width, height))
    # # print("OKOK", tf.TensorShape((nvoxgrids, nmatrices, width, height)))
    return img
    # pixels = tf.reshape(img, img_shape)
    # mask = t14 > t04
    # # print(mask.reshape())
    # return pixels,
    # return tf.select(mask.reshape(img_shape), pixels, tf.ones_like(pixels)), rd, ro, tn_x, tf.ones(img_shape), orig, voxels


def render_fwd_f(inputs):
    voxels = inputs['voxels']
    options = {}
    width = options['width'] = 128
    height = options['height'] = 128
    res = options['res'] = 32
    nsteps = options['nsteps'] = 3
    nvoxgrids = options['nvoxgrids'] = 1
    nviews = options['nviews'] = 1
    rotation_matrices = rand_rotation_matrices(nviews)
    out_img = gen_img(voxels, rotation_matrices, width, height, nsteps, res)
    outputs = {"out_img": out_img}
    return outputs


def render_gen_graph(g, batch_size):
    """Generate a graph for the rendering function"""
    nvoxgrids = 1
    res = 32
    with g.name_scope("fwd_g"):
        voxels = tf.placeholder(floatX(), name="voxels",
                                shape=(batch_size, nvoxgrids*res*res*res))
                                # shape=(nvoxgrids, res, res, res))
        inputs = {"voxels": voxels}
        outputs = render_fwd_f(inputs)
        return {"inputs": inputs, "outputs": outputs}


def draw_random_voxels():
    sess = tf.Session()

    # load data
    voxels_path = os.path.join(os.environ['DATADIR'],
                               'ModelNet40', 'alltrain32.npy')
    voxel_grids = np.load(voxels_path)
    rand_voxel_id = np.random.randint(0, voxel_grids.shape[0])
    res = 32
    voxels = results['inputs']['voxels']
    width = 32
    height = 32
    voxel = voxel_grids[rand_voxel_id].reshape(1, res, res, res)

    # Run the graph
    output_img = sess.run(out_img_tensor, feed_dict={voxels: voxel})
    # print(output_img.shape)
    plt.imshow(output_img.reshape(width, height))
    plt.show()
    sess.close()


def test_render_graph():
    g = tf.get_default_graph()
    batch_size = 4
    results = render_gen_graph(g, batch_size)
    out_img_tensor = results['outputs']['out_img']
    arrow_renderer = graph_to_arrow([out_img_tensor])
    inv_renderer = invert(arrow_renderer)
    import pdb; pdb.set_trace()


def gen_data(batch_size):
    """Generate data for training"""
    graph = tf.Graph()
    with graph.as_default():
        inputs, outputs = getn(render_gen_graph(graph, batch_size), 'inputs', 'outputs')
        inputs = [inputs['voxels']]
        outputs = [outputs['out_img']]
        input_data = [np.random.rand(*(i.get_shape())) for i in inputs]
        sess = tf.Session()
        output_data = sess.run(outputs, feed_dict=dict(zip(inputs, input_data)))
        sess.close()
    return {'inputs': input_data, 'outputs': output_data}


def pi_supervised(options):
    """Neural network enhanced Parametric inverse! to do supervised learning"""
    tf.reset_default_graph()
    g = tf.get_default_graph()
    batch_size = options['batch_size']
    results = render_gen_graph(g, batch_size)
    out_img_tensor = results['outputs']['out_img']
    arrow_renderer = graph_to_arrow([out_img_tensor])
    inv_arrow = inv_fwd_loss_arrow(arrow_renderer)
    right_inv = unparam(inv_arrow)
    sup_right_inv = supervised_loss_arrow(right_inv)
    # Get training and test_data
    train_data = gen_data(batch_size)
    test_data = gen_data(1024)

    # Have to switch input from output because data is from fwd model
    train_input_data = train_data['outputs']
    train_output_data = train_data['inputs']
    test_input_data = test_data['outputs']
    test_output_data = test_data['inputs']
    num_params = get_tf_num_params(right_inv)
    print("Number of params", num_params)
    supervised_train(sup_right_inv,
                     train_input_data,
                     train_output_data,
                     test_input_data,
                     test_output_data,
                     callbacks=[],
                     options=options)


# Benchmarks
from metrics.generalization import test_generalization
def generalization_bench():
    options = handle_options('voxel_render', sys.argv[1:])
    sfx = gen_sfx_key(('nblocks', 'block_size'), options)
    options['sfx'] = sfx
    options['description'] = "Voxel Generalization Benchmark"
    test_generalization(pi_supervised, options)


if __name__ == "__main__":
    generalization_bench()
    test_render_graph()
