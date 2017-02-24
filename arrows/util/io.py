import getopt
import os
import csv
import time
import sys
from optparse import (OptionParser,BadOptionError,AmbiguousOptionError)


def save_params(fname, params):
    f = open(fname, 'w')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([key, value])
    f.close()


def save_dict_csv(fname, params):
    f = open(fname, 'w')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([str(key), str(value)])
    f.close()


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directlds a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


class PassThroughOptionParser(OptionParser):
    """
    An unknown option pass-through implementation of OptionParser.

    When unknown arguments are encountered, bundle with largs and try again,
    until rargs is depleted.

    sys.exit(status) will still be called if a known argument is passed
    incorrectly (e.g. missing arguments or bad argument types, etc.)
    """
    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                OptionParser._process_args(self, largs, rargs, values)
            except (BadOptionError, AmbiguousOptionError) as e:
                largs.append(e.opt_str)


def handle_args(argv, cust_opts):
    """Handle getting options from command liner arguments"""
    custom_long_opts = ["%s=" % k for k in cust_opts.keys()]
    cust_double_dash = ["--%s" % k for k in cust_opts.keys()]
    parser = PassThroughOptionParser()
    parser.add_option('-l', '--learning_rate', dest='learning_rate',nargs=1, type='int')

    # Way to set default values
    # some flags affect more than one thing
    # some things need to set otherwise everything goes to shit
    # some things need to be set if other things are set

    long_opts = ["params_file=", "learning_rate=", "momentum=", "update=",
                 "description=", "template=", "batch_size="] + custom_long_opts
    options = {'params_file': '', 'learning_rate': 0.1, 'momentum': 0.9,
               'load': False, 'update': 'momentum', 'description': '',
               'template': 'res_net', 'batch_size': 128}
    help_msg = """-p <paramfile> -l <learning_rate> -m <momentum> -u <update algorithm> -d
                  <job description> -t <template>"""
    try:
        opts, args = getopt.getopt(argv, "hp:l:m:u:d:t:", long_opts)
    except getopt.GetoptError:
        print("invalid options")
        print(help_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_msg)
            sys.exit()
        elif opt in ("-p", "--params_file"):
            options['params_file'] = arg
            options['load'] = True
        elif opt in ("-l", "--learning_rate"):
            options['learning_rate'] = float(arg)
        elif opt in ("-m", "--momentum"):
            options['momentum'] = float(arg)
        elif opt in ("-u", "--update"):
            if arg in ['momentum', 'adam', 'rmsprop']:
                options['update'] = arg
            else:
                print("update must be in ", ['momentum', 'adam', 'rmsprop'])
                print(help_msg)
                sys.exit()
        elif opt in ("-d", "--description"):
            options['description'] = arg
        elif opt in ("-t", "--template"):
            options['template'] = arg
        elif opt in cust_double_dash:

            opt_key = opt[2:]  # remove --
            cust = cust_opts[opt_key]
            assert len(cust) == 2
            f, default = cust
            options[opt_key] = f(arg)

    # import pdb; pdb.set_trace()

    # add defaults back
    for (key, val) in cust_opts.items():
        if key not in options:
            options[key] = val[-1]

    print(options)
    return options

def mk_dir(sfx='',
           dirname=None,
           datadir = os.environ['DATADIR']):
    "Create directory with timestamp"
    if dirname is None:
        newdirname = str(time.time()) + sfx
    else:
        newdirname = dirname
    full_dir_name = os.path.join(datadir, "rf", newdirname)
    print("Data will be saved to", full_dir_name)
    os.mkdir(full_dir_name)
    return full_dir_name
