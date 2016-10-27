"""Array Operations"""
from reverseflow.arrows.primitivearrow import PrimitiveArrow


class Gather(PrimitiveArrow):
    """
    Gather slices from `params` according to `indices`.

    `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    Produces an output tensor with shape `indices.shape + params.shape[1:]`,
    where:

    for i = 0...rank(indices)
    output[i] = params[indices[i]]



    ```python
        # Scalar indices
        output[:, ..., :] = params[indices, :, ... :]

        # Vector indices
        output[i, :, ..., :] = params[indices[i], :, ... :]

        # Higher rank indices
        output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
    ```

    If `indices` is a permutation and `len(indices) == params.shape[0]` then
    this operation will permute `params` accordingly.

    Args:
      params: A `Tensor`.
      indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      validate_indices: An optional `bool`. Defaults to `True`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `params`.

    """

    def __init__(self):
        name = 'Gather'
        super().__init__(n_in_ports=2, n_out_ports=1, name=name)
