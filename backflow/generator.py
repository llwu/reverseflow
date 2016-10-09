from bf.util import *
import tensorflow as tf
import numpy as np
import random
import pdb


class Choice:
    def __init__(self, transform, prob, **kwargs):
        assert prob > 0
        self.transform = transform
        self.prob = prob
        self.kwargs = kwargs


## Predicates
## ==========
def check_empty(g):
    return g.get_operations() == []

def two_equiv_tensors(tensor_groups):
    """Get two tensors with same shape and dtype"""
    for group in tensor_groups.values():
        if len(group) > 1:
            return True
    return False

## Transforms
## ==========
def create_var(g, dtype, shape):
    tf.placeholder(dtype=dtype, shape=shape)
    return False

def create_const(g, const_gen):
    tf.constant(const_gen())
    return False

def apply_op(g, op, args):
    op(*args)
    return False

def stop_signal(g):
    return True

## suggestions
## ===========
def create_vars(g):
    n = num_ph(g)
    if n == 0:
        return [Choice(create_var, 1.0, dtype=tf.float32, shape=(128,128))]
    else:
        return [Choice(create_var, 1.0, dtype=tf.float32, shape=(128,128)),
                Choice(create_const, 0.5, const_gen=lambda: np.random.rand(10, 10))]

def maybe_stop(g):
    if len(all_tensors_namescope(g, 'fwd_g')) > 10.0:
        return [Choice(stop_signal, 1.0)]
    else:
        return []

def apply_elem_op(g):
    tensors = all_tensors_namescope(g, 'fwd_g')
    valid_tensors = []
    for t in tensors:
        if in_namescope(t, "fwd_g") and (t.op.type == "placeholder" or t.op.type == "Identity"):
            valid_tensors.append(t)
        elif in_namescope(t, "random_graph"):
            valid_tensors.append(t)
    pdb.set_trace()
    tensor_groups = group_equiv_tensors(valid_tensors)
    if two_equiv_tensors(tensor_groups):
        ops = [tf.add, tf.sub, tf.mul]
        for v in tensor_groups.values():
            print("V is", len(v))
            if len(v) > 1:
                a, b  = np.random.choice(v, (2,), replace=False)
                op = np.random.choice(ops)
                return [Choice(apply_op, 2.0, op=op, args=(a, b))]

        assert False
    else:
        return []

def gen_graph(g, suggestions, max_iterations=1000):
    """
    Generate a tensorlow graph
    g :: tf.Graph - graph to append to, if None creates new graph
    """
    np.random.seed(0)
    random.seed(0)
    print("Generating Graph")
    with g.as_default():
        with g.name_scope("random_graph"):
            for i in range(max_iterations):
                # pdb.set_trace()
                choices = []
                for suggest in suggestions:
                    choices = choices + suggest(g)
                weights = [c.prob for c in choices]
                print(weights)
                probs = weights / np.sum(weights)
                curr_choice = np.random.choice(choices, p=probs)
                print(i," ", curr_choice.prob)
                stop_now = curr_choice.transform(g, **curr_choice.kwargs)
                if stop_now:
                    print("Caught stop signal, stopping")
                    break
    print(summary(g))
    detailed_summary(g)

    return g

# g = tf.Graph()
# gen_graph(g, [create_vars, maybe_stop, apply_elem_op])
# print(summary(g))
# writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', g)
