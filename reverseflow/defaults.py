# """Default mappings between parametric inverses and inverses"""
# from reverseflow.arrows.arrow import Arrow
# from reverseflow.util.mapping import Bimap, ImageBimap
# from reverseflow.arrows.primitive.math_arrows import (AddArrow, SubArrow,
#                                                       MulArrow)
# from reverseflow.inv_primitives.inv_math_arrows import InvAddArrow, InvSubArrow
#
# arrows = [AddArrow, SubArrow, MulArrow]
# inv_arrows = [InvAddArrow]
#
# """A relation between forward arrows and (possibly many) inverses"""
# INV_TO_FWD = ImageBimap()  # type: ImageBimap[Arrow, Arrow]
#
# for inv_arrow in inv_arrows:
#     a = inv_arrow # type: int
#     INV_TO_FWD.add(a, inv_arrow.inverse_of())
#
#
# AddArrow, InvAddArrow, {}
# AddArrow, SubArrow, {0: 1}
# AddArrow, SubArrow, {1: 1}
#
# SubArrow, InvSubArrow, {}
# SubArrow, SubArrow, {0: 0}
# AddArrow, AddArrow, {1: 0}
#
#
#
#
# List[int]
#
# mapping = Dict[ArrowClass]
