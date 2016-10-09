import numpy as np

def exponential(w, l):
    """Exponentially distributed random variable"""
    return - np.log(1-w)/l

def trunc_exp(phi):
    a = exponential(phi, 1)
    b = -a
    c = b + 10
    print(c)
    theta_1 = c
    theta_2 = 5 - theta_1
    inv_lt = 10 - theta_1
    inv_gt = 5 + theta_2
    return (inv_lt, inv_gt)
