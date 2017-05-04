"""Helpers for Dealing with Voxels"""
import numpy as np
import os
import math
from functools import reduce


def model_net_40(voxels_path=os.path.join(os.environ['DATADIR'],
                                          'ModelNet40',
                                          'alltrain32.npy')):
    voxel_grids = np.load(voxels_path) / 255.0
    return voxel_grids


def model_net_40_grads(voxels_path=os.path.join(os.environ['DATADIR'],
                                                'ModelNet40',
                                                'alltrain32grads.npz')):
    return np.load(voxels_path)['arr_0']


def model_net_fake(data_size=1024):
    return np.random.rand(data_size, 32, 32, 32)


def voxel_indices(voxels, limit, missing_magic_num=-1):
    """
    Convert voxel data_set (n, 32, 32, 32) to (n, 3, m)
    """
    n, x, y, z = voxels.shape
    output = np.ones((n, 4, limit))
    output = output * missing_magic_num

    # obtain occupied voxels
    for v in range(len(voxels)):
        voxel = voxels[v]
        x_list, y_list, z_list = np.where(voxel)
        assert len(x_list) == len(y_list)
        assert len(y_list) == len(z_list)

        # fill in output tensor
        npoints = min(limit, len(x_list))
        output[v][0][0:npoints] = x_list[0:npoints]
        output[v][1][0:npoints] = y_list[0:npoints]
        output[v][2][0:npoints] = z_list[0:npoints]
        output[v][3][0:npoints] = voxel[x_list, y_list, z_list][0:npoints]

    output = np.transpose(output, [0, 2, 1]) # switch limit and coords

    return output


def indices_voxels(indices, grid_x=32, grid_y=32, grid_z=32):
    """Convert indices representation into voxel grid"""
    indices = indices.astype('int')
    nvoxels = indices.shape[0]
    # indices = np.transpose(indices, (0, 2, 1))
    voxel_grid = np.zeros((nvoxels, grid_x, grid_y, grid_z))
    for i in range(nvoxels):
        for j in range(indices.shape[1]):
            x = indices[i, j, 0]
            y = indices[i, j, 1]
            z = indices[i, j, 2]
            d = indices[i, j, 3]
            if 0 <= x < grid_x and 0 <= y < grid_y and 0 <= z < grid_z:
                voxel_grid[i, x, y, z] = d
            else:
                break

    return voxel_grid


def rand_rotation_matrix(deflection=1.0, randnums=None, floatX='float32'):
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
    return np.array(M, dtype=floatX)

def show_voxel_grid(grid):
    from mayavi import mlab
    # from mayavi import mlab
    """Vizualise voxel grid with mlab
    x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    s = np.sin(x*y*z)/(x*y*z)
    """
    mlab.pipeline.volume(mlab.pipeline.scalar_field(grid), vmin=0, vmax=0.8)
    mlab.show()


def cast(x, typ):
    """Cast an array into a particular type"""
    return x.astype(typ)


def get_indices(pos, res, tnp=np):
    """
    pos: (npos, 3)
    returns (npos, 3)
    """
    voxel_indices = tnp.floor(pos * res)
    clamped = tnp.clip(voxel_indices, 0, res-1)
    p_int =  cast(clamped, 'int32',)
    indices = tnp.reshape(p_int, (-1, 3))
    return indices


def gen_voxel_grid(res):
    "returns (res, res, res, 3) array"
    return np.transpose(np.mgrid[0:1:complex(res),0:1:complex(res),0:1:complex(res)],(1,2,3,0))


def sample_volume(voxels, indices, res):
    """Samples voxels as pos positions
    voxels : (nvoxgrids, res, res, res)
    pos : (npos, 3)
    returns : (nvoxgrids, npos)"""
    indices = np.clip(indices, 0, res - 1)
    return voxels[:, indices[:, 0], indices[:, 1], indices[:, 2]]


def xyz_diff(voxels, pos, xyz_delta_pos, xyz_delta_neg, res, tnp=np):
    """Compute finite differences in a particular dimension"""
    xyz1 = sample_volume(voxels, pos + xyz_delta_pos, res)
    xyz2 = sample_volume(voxels, pos + xyz_delta_neg, res)
    return xyz1 - xyz2   # nbatch * width * height * depth


def normalise(x, axis, eps=1e-9):
    return x / ((np.expand_dims(np.linalg.norm(x, axis=axis), axis)) + eps)


def compute_gradient(pos, voxels, res, n=1):
    """
    Compute gradient of voxels using finite differences
    Args:
      pos : (npos, 3)
      voxels : (num_voxels, res, res, res)
      n: number of voxels
    returns : (num_voxels, npos, 3)
    """
    x_diff = xyz_diff(voxels, pos, np.array([n, 0, 0]), np.array([-n, 0, 0]), res)
    y_diff = xyz_diff(voxels, pos, np.array([0, n, 0]), np.array([0, -n, 0]), res)
    z_diff = xyz_diff(voxels, pos, np.array([0, 0, n]), np.array([0, 0, -n]), res)
    gradients = np.stack([x_diff, y_diff, z_diff], axis=2)
    return normalise(gradients, axis=2)


def full_compute_grad(voxels, res, n=1):
    vox_grid = gen_voxel_grid(res)
    pos = get_indices(vox_grid, 32)
    return compute_gradient(pos, voxels, res, n=n)

# FIXME: This batch should be abstracted to another funciton
def batch_compute_grad(voxels, res, batch_size, n=1):
    vox_grid = gen_voxel_grid(res)
    pos = get_indices(vox_grid, 32)
    niters = math.ceil(len(voxels) // batch_size)
    grad_batch = []
    for i in range(niters):
        lb = i * batch_size
        ub = min((i * batch_size) + batch_size, len(voxels))
        voxel_batch = voxels[lb:ub]
        grad_batch.append(compute_gradient(pos, voxel_batch, res, n=n))
    return np.concatenate(grad_batch, axis=0)


def cartesian_product(arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def cube_filter(voxels, res, n=1):
    """Smooth gradients
    Take nvoxels,res,res,res voxels and return something of the same size"""
    indices_range = np.arange(res)
    indices = cartesian_product([indices_range, indices_range, indices_range])
    x_zero = indices[:, 0]
    y_zero = indices[:, 1]
    z_zero = indices[:, 2]
    x_neg = np.clip(indices[:, 0] - n, 0, res - 1)
    x_add = np.clip(indices[:, 0] + n, 0, res - 1)
    y_neg = np.clip(indices[:, 1] - n, 0, res - 1)
    y_add = np.clip(indices[:, 1] + n, 0, res - 1)
    z_neg = np.clip(indices[:, 2] - n, 0, res - 1)
    z_add = np.clip(indices[:, 2] + n, 0, res - 1)
    v_x_add = voxels[:, x_add, y_zero, z_zero]
    v_x_neg = voxels[:, x_neg, y_zero, z_zero]
    v_y_add = voxels[:, x_zero, y_add, z_zero]
    v_y_neg = voxels[:, x_zero, y_neg, z_zero]
    v_z_add = voxels[:, x_zero, y_zero, z_add]
    v_z_neg = voxels[:, x_zero, y_zero, z_neg]

    voxels_flat = voxels.reshape((-1, res**3))
    voxels_mean = (v_x_add + v_x_neg + v_y_add + v_y_neg + v_z_add + v_z_neg + voxels_flat)/7.0
    return voxels_mean.reshape(voxels.shape)


def gdotl(light_dir, vox_grads, res, batch_size, nfilters=5):
    """
    Compute dot product of light vector with filted gradient vector
    light_dir: vector direction of light, e.g, np.array([[[0, 1, 1]]])
    """
    gdotl = np.sum((light_dir * vox_grads), axis=2)
    gdotl_cube = gdotl.reshape((batch_size, res, res, res))
    # Filter the gradients
    for i in range(1, nfilters):
      gdotl_cube = cube_filter(gdotl_cube, res, i)
    gdotl_cube = np.maximum(0, gdotl_cube)
    return gdotl_cube


# vox_grads = model_net_40_grads()
# light_dir = np.array([[[0, 1, 1]]])
# smelly = gdotl(light_dir, vox_grads, 32, 10)
