"""Functions common for examples"""
import sys
from arrows.util.io import *

import tensortemplates.res_net as res_net
import tensortemplates.conv_res_net as conv_res_net

def stringy_dict(d):
    out = ""
    for (key, val) in d.items():
        if val is not None and val is not '':
            out = out + "%s_%s__" % (str(key), str(val))
    return out

def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


template_module = {'res_net': res_net, 'conv_res_net': conv_res_net}

def boolify(x):
    if x in ['0', 0, False, 'False', 'false']:
        return False
    elif x in ['1', 1, True, 'True', 'true']:
        return True
    else:
        assert False, "couldn't convert to bool"


def default_kwargs():
    """Default kwargs"""
    options = {}
    options['learning_rate'] = (float, 0.1)
    options['update'] = (str, 'momentum')
    options['params_file'] = (str, 28)
    options['momentum'] = (float, 0.9)
    options['description'] = (str, "")
    options['batch_size'] = (int, 128)
    options['save_every'] = (int, 100)
    options['compress'] = (boolify, 0,)
    options['num_iterations'] = (int, 1000)
    options['save'] = (boolify, True)
    options['template'] = (str, 'res_net')
    options['train'] = (boolify, True)
    return options


def handle_options(name, argv):
    """Parse options from the command line and populate with defaults"""
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    # Get default options
    options = default_kwargs()
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template

    # Get template specific options
    template_kwargs = template_module[options['template']].kwargs()
    options.update(template_kwargs)
    options['name'] = (str, name)
    options = handle_args(argv, options)
    options['template'] = template_module[options['template']].template
    return options
