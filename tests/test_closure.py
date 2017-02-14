from reverseflow.invert import invert

from test_arrows import test_twoxyplusx

def test_closure():
    arrow = test_twoxyplusx()
    inv_arrow = invert(arrow)
    return inv_arrow


if __name__ == '__main__':
    inv_arrow = test_closure()
    import pdb; pdb.set_trace()
