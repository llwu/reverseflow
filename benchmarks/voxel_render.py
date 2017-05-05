""" (Inverse) Rendering"""
import os
import signal
import sys
import pdb

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from arrows.apply.apply import apply, apply_backwards
from arrows.apply.propagate import propagate
from arrows.config import floatX
from arrows.port_attributes import is_param_port, is_error_port
from arrows.transform.eliminate_gather import eliminate_gathernd
from arrows.util.misc import getn
from arrows.util.viz import show_tensorboard, show_tensorboard_graph
from benchmarks.common import handle_options
from metrics.generalization import test_generalization
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.common import get_tf_num_params
from reverseflow.train.loss import inv_fwd_loss_arrow, supervised_loss_arrow
from reverseflow.train.supervised import supervised_train
from reverseflow.train.unparam import unparam

from voxel_helpers import model_net_40, model_net_40_gdotl, rand_rotation_matrices

# For repeatability
# STD_ROTATION_MATRIX = rand_rotation_matrices(nviews)
STD_ROTATION_MATRIX = np.array([[[0.94071758, -0.33430171, -0.05738258],
                                 [-0.33835238, -0.91297877, -0.2280076],
                                 [0.02383425, 0.2339063, -0.97196698]]])

# Genereate values in raster space, x[i,j] = [i,j]
def gen_fragcoords(width, height):
    """Create a (width * height * 2) matrix, where element i,j is [i,j]
       This is used to generate ray directions based on an increment"""
    raster_space = np.zeros([width, height, 2], dtype=floatX())
    for i in range(width):
        for j in range(height):
            raster_space[i, j] = np.array([i, j], dtype=floatX()) + 0.5
    return raster_space


# Append an image filled with scalars to the back of an image.
def stack(intensor, width, height, scalar):
    scalars = np.ones([width, height, 1], dtype=floatX()) * scalar
    return np.concatenate([intensor, scalars], axis=2)


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
    aspect_ratio = resolution[0] / resolution[1]
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

# 1. possibly some constraints on the options, i.e. a struct ratehr than anything
# goes dictionary
# 2. want to be able to call the function independently from creating a struct
# 3. I want to be able to pass the struct along easily without its members being lost



def gen_img(voxels, gdotl_cube, rotation_matrix, options):
    """Renders `batch_size` voxel grids
    Args:
      voxels : (batch_size, res, res, res)
      rotation_matrix : (m, 4)
      width: width in pixels of rendered image
      height: height in pixels of rendered image
      nsteps: number of points along each ray to sample voxel grid
      res: voxel resolution 'voxels' should be res * res * res
      batch_size: number of voxels to render in batch
      gdot_cube: dot product of gradient and light, can be computed offline
                 used only in phond shading
      phong: do phong shading

    Returns:
      (n, m, width, height) - from voxel data from functions in voxel_helpers
    """
    width, height, nsteps, res, batch_size, phong, density = getn(options,
      'width', 'height', 'nsteps', 'res', 'batch_size', 'phong', 'density')

    if phong:
      if gdotl_cube is None:
        raise(ValueError("Must provide gdotl_cube for phong rendering"))

    raster_space = gen_fragcoords(width, height)
    rd, ro = make_ro(rotation_matrix, raster_space, width, height)
    a = 0 - ro  # c = 0
    b = 1 - ro  # c = 1
    nmatrices = rotation_matrix.shape[0]
    tn = np.reshape(a, (nmatrices, 1, 1, 3)) / rd
    tff = np.reshape(b, (nmatrices, 1, 1, 3)) / rd
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
    t04 = t04 * 1.001
    t14 = t14 * 0.999

    left_over = np.ones((batch_size, nmatrices * width * height,))
    step_size = (t14 - t04) / nsteps
    orig = np.reshape(ro, (nmatrices, 1, 1, 3)) + rd * np.reshape(t04,(nmatrices, width, height, 1))
    xres = yres = res

    orig = np.reshape(orig, (nmatrices * width * height, 3))
    rd = np.reshape(rd, (nmatrices * width * height, 3))
    step_sz = np.reshape(step_size, (nmatrices * width * height, 1))
    step_sz_flat = step_sz.reshape(nmatrices * width * height)

    # For batch rendering, we treat each voxel in each voxel independently,
    nrays = width * height
    x = np.arange(batch_size)
    x_tiled = np.repeat(x, nrays)

    for i in range(nsteps):
        # Find the position (x,y,z) of ith step
        pos = orig + rd * step_sz * i

        # convert to indices for voxel cube
        voxel_indices = np.floor(pos * res)
        pruned = np.clip(voxel_indices, 0, res - 1)
        p_int = pruned.astype('int64')
        indices = np.reshape(p_int, (nmatrices * width * height, 3))

        # convert to indices in flat list of voxels
        flat_indices = indices[:, 0] + res * (indices[:, 1] + res * indices[:, 2])

        # tile the indices to repeat for all elements of batch
        tiled_indices = np.tile(flat_indices, batch_size)
        batched_indices = np.transpose([x_tiled, tiled_indices])
        batched_indices = batched_indices.reshape(batch_size, len(flat_indices), 2)
        attenuation = tf.gather_nd(voxels, batched_indices)
        if phong:
            grad_samples = tf.gather_nd(gdotl_cube, batched_indices)
            attenuation = attenuation * grad_samples
        left_over = left_over * tf.exp(-attenuation * density * step_sz_flat)

    img = left_over
    return img


def default_options():
    """Default options for rendering"""
    return {'batch_size': 8,
            'width': 128,
            'height': 128,
            'res': 32,
            'nsteps': 100,
            'nviews': 1,
            'density': 10.0,
            'phong': False}


def render_gen_graph(options):
    """Generate a graph for the rendering function"""
    res = options.get('res')
    batch_size = options.get('batch_size')
    phong = options.get('phong')
    nviews = options.get('nviews')

    nvoxels = res * res * res

    with tf.name_scope("fwd_g"):
        voxels = tf.placeholder(floatX(), name="voxels",
                                shape=(batch_size, nvoxels))

        if phong:
            gdotl_cube = tf.placeholder(floatX(), name="gdotl",
                                        shape=(batch_size, nvoxels))
        else:
            gdotl_cube = None

        rotation_matrices = STD_ROTATION_MATRIX
        out_img = gen_img(voxels, gdotl_cube, rotation_matrices, options)
        return {'voxels': voxels,
                'gdotl_cube': gdotl_cube,
                'out_img': out_img}


def test_invert_render_graph(options):
    out_img = render_gen_graph(options)['out_img']
    arrow_renderer = graph_to_arrow([out_img_tensor], name="renderer")
    inv_renderer = invert(arrow_renderer)
    return arrow_renderer, inv_renderer


def get_param_pairs(inv, voxel_grids, batch_size, n, port_attr=None,
                    pickle_to=None):
    """Pulls params from 'forward' runs. FIXME: mutates port_attr."""
    if port_attr is None:
        port_attr = propagate(inv)
    shapes = [port_attr[port]['shape'] for port in inv.out_ports() if not is_error_port(port)]
    params = []
    inputs = []
    for i in range(n):
        rand_voxel_id = np.random.randint(0, voxel_grids.shape[0], size=batch_size)
        input_data = [voxel_grids[rand_voxel_id].reshape(shape).astype(np.float32) for shape in shapes]
        inputs.append(input_data)
        params_bwd = apply_backwards(inv, input_data, port_attr=port_attr)
        params_list = [params_bwd[port] for port in inv.in_ports() if is_param_port(port)]
        params.append(params_list)
    if pickle_to is not None:
        with open(pickle_to, 'wb') as f:
            pickle.dump((inputs, params), f)
    return inputs, params


def plot_batch(image_batch):
  """Plot a batch of images"""
  batch_size = len(image_batch)
  for i in range(batch_size):
      img = image_batch[i].reshape(128, 128)
      plt.imshow(img, cmap='gray')
      plt.ioff()
      plt.show()


def inv_viz_allones(voxel_grids, options, batch_size):
    """Invert voxel renderer, run with all 1s as parameters, visualize"""
    arrow, inv = test_render_graph(options, batch_size=batch_size)
    info = propagate(inv)
    inputs, params = get_param_pairs(inv, voxel_grids, batch_size, 1,
                                     port_attr=info)
    outputs = apply(arrow, inputs[0])
    output_list = [outputs[0]] + params[0]
    recons = apply(inv, output_list)[0]
    recons_fwd = apply(arrow, [recons])
    for i in range(batch_size):
        img_A = outputs[0][i].reshape(128, 128)
        padding = np.zeros((128, 16))
        img_B = recons_fwd[0][i].reshape(128, 128)
        plot_image = np.concatenate((img_A, padding, img_B), axis=1)
        plt.imshow(plot_image, cmap='gray')
        plt.ioff()
        plt.show()


def render_rand_voxels(voxels_data, gdotl_cube_data, options):
    """Render `batch_size` randomly selected voxels for voxel_grids"""
    batch_size = options.get('batch_size')
    graph = tf.Graph()
    with graph.as_default():
        voxels, gdotl_cube, out_img = getn(render_gen_graph(options),
                                           'voxels', 'gdotl_cube', 'out_img')
        rand_id = np.random.randint(len(voxels_data), size=batch_size)
        input_voxels = [voxels_data[rand_id[i]].reshape(voxels[i].get_shape()) for i in range(batch_size)]

        sess = tf.Session()
        feed_dict = {voxels: input_voxels}
        if options['phong']:
            input_gdotl_cube = [gdotl_cube_data[rand_id[i]].reshape(gdotl_cube[i].get_shape()) for i in range(batch_size)]
            feed_dict[gdotl_cube] = input_gdotl_cube
        out_img_data = sess.run(out_img, feed_dict=feed_dict)
        sess.close()
    return {'input_voxels': input_voxels,
            'out_img_data': out_img_data}


def main():
    def debug_signal_handler(signal, frame):
        pdb.set_trace()
    signal.signal(signal.SIGINT, debug_signal_handler)
    voxel_grids = model_net_40()
    inv_viz_allones(voxel_grids, options, batch_size=8)
    # generalization_bench()


def test_renderer():
    options = default_options()
    voxel_grids = model_net_40()
    if options['phong']:
        gdotl_cube_data = model_net_40_gdotl()
    else:
        gdotl_cube_data = None
    inp_out = render_rand_voxels(voxel_grids, gdotl_cube_data, options)
    plot_batch(inp_out['out_img_data'])


if __name__ == "__main__":
    test_renderer()
    # main()

# Benchmarking
def pi_supervised(options):
    """Neural network enhanced Parametric inverse! to do supervised learning"""
    tf.reset_default_graph()
    g = tf.get_default_graph()
    batch_size = options['batch_size']
    results = render_gen_graph(g, batch_size)
    out_img_tensor = results['outputs']['out_img']
    arrow_renderer = graph_to_arrow([out_img_tensor])
    inverted = invert(arrow_renderer)
    inv_arrow = inv_fwd_loss_arrow(arrow_renderer, inverted)
    right_inv = unparam(inv_arrow)
    sup_right_inv = supervised_loss_arrow(right_inv)
    # Get training and test_data
    train_data = render_rand_voxels(batch_size)
    test_data = render_rand_voxels(1024)

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


def generalization_bench():
    """Benchmark generalization ability of inverse graphics methods"""
    options = handle_options('voxel_render', sys.argv[1:])
    sfx = gen_sfx_key(('nblocks', 'block_size'), options)
    options['sfx'] = sfx
    options['description'] = "Voxel Generalization Benchmark"
    test_generalization(pi_supervised, options)
