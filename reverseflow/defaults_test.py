"""Default mappings between parametric inverses and inverses"""
from reverseflow.arrows.arrow import Arrow
from reverseflow.util.mapping import Bimap, ImageBimap
from reverseflow.arrows.primitive.math_arrows import (AddArrow, SubArrow,
                                                      MulArrow)
from reverseflow.inv_primitives.inv_math_arrows import InvAddArrow
from reverseflow.defaults import INV_TO_FWD

arrows = [AddArrow, SubArrow, MulArrow]

# for arrow in arrows:
#     keys = INV_TO_FWD.keys()
#     assert arrow in keys, "No inverse defined for arrow:%s", % arrow
