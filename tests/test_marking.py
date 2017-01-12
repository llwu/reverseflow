"""Tests propagation of known ports."""
import pdb

from arrows.marking import mark
from arrows.primitive.math_arrows import AddArrow
from arrows.compositearrow import CompositeArrow
from arrows.sourcearrow import SourceArrow
from test_arrows import test_mixed_knowns


def manual_inspection():
    """Manually inspect output with PDB."""
    arrow = test_mixed_knowns()
    marks = mark(arrow)
    pdb.set_trace()
    return marks


if __name__ == '__main__':
    manual_inspection()
