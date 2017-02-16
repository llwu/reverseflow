"""Functions common for examples"""
import sys
from arrows.util.io import *

from pdt.util.misc import stringy_dict
import tensortemplates.res_net as res_net
import tensortemplates.conv_res_net as conv_res_net


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

def handle_options(adt, argv):
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    options = {}
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template
    template_kwargs = template_module[options['template']].kwargs()
    options.update(template_kwargs)
    options['train'] = (boolify, 1)
    options['nitems'] = (int, 3)
    options['width'] = (int, 28)
    options['height'] = (int, 28)
    options['num_epochs'] = (int, 100)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['compress'] = (boolify, 0)
    options['compile_fns'] = (boolify, 1)
    options['save_params'] = (boolify, 1)
    options['adt'] = (str, adt)
    options = handle_args(argv, options)
    options['template'] = template_module[options['template']].template
    return options
