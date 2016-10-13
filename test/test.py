from rf.compare import compare
import sys
import getopt
import tensorflow as tf
import numpy as np
from rf.util import *
from rf.templates.res_net import res_net_template_dict
from rf.generator import *
import random

for i in range(4):
    g = tf.Graph()
    gen_graph(g, [create_vars, maybe_stop, apply_elem_op])
    writer = tf.train.SummaryWriter('/home/zenna/repos/inverse/log', g)
