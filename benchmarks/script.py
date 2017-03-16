"""Run the openmind script for parallel hyperparameter search"""
import sys
import ast
from common import *
from arrows.util.io import handle_args
from reverseflow.train.callbacks import save_options, save_every_n, save_everything_last
from benchmarks.linkage_kinematics import robo_tensorflow
import tensortemplates.res_net as res_net
import tensortemplates.conv_res_net as conv_res_net
import numpy as np

method_map = {'pi_supervised': pi_supervised,
              'nn_supervised': nn_supervised,
              'pi_reparam': pi_reparam,
              'save_options': save_options,
              'save_every_n': save_every_n,
              'save_everything_last': save_everything_last,
              'gen_rand_data': gen_rand_data,
              'robo_tensorflow': robo_tensorflow}

template_module = {'res_net': res_net, 'conv_res_net': conv_res_net}

def parse_dict(string_d):
    parsed_dict = ast.literal_eval(string_d)
    # special case for the field 'template'
    if 'template' in parsed_dict:
        assert 'template_name' in parsed_dict
        parsed_dict['template'] = template_module[parsed_dict['template_name']].template
    for key, value in parsed_dict.items():
        if key == 'template':
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            new_val = []
            for val in value:
                if val in method_map:
                    new_val.append(method_map[val])
                else:
                    new_val.append(val)
            parsed_dict[key] = new_val
        elif value in method_map:
            parsed_dict[key] = method_map[value]
    return parsed_dict

if __name__ == "__main__":
    options = {}
    options = handle_args(sys.argv[1:], options)
    assert 'run' in options and 'options' in options
    model_options = parse_dict(options['options'])
    runner = method_map[options['run']]
    runner(model_options)
