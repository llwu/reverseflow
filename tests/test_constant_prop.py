"""Tests propagation of known ports."""
import pdb

from arrows.apply.constants import propagate_constants
from test_arrows import test_mixed_knowns


def manual_inspection():
    """Manually inspect output with PDB."""
    arrow = test_mixed_knowns()
    marks = propagate_constants(arrow)
    pdb.set_trace()
    return marks


if __name__ == '__main__':
    manual_inspection()
