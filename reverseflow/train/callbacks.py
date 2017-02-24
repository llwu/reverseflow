"""Generalization Test"""

# So its better to prevent train_loop from becoming this huge mess
# and instead pass in a callback
# a lens would be even better but there you go, we dont have that
import os
import pickle
import csv

def save_it_csv(params, fname):
    f = open(fname, 'w')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([str(key), str(value)])
    f.close()


def pickle_it(data, save_dir):
    with open(save_dir, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def save_callback(fetch_data,
                  feed_dict,
                  i: int,
                  compress=False,
                  pfx = '',
                  **kwargs):
    sess = kwargs['sess']
    save = kwargs['save']
    if  save:
        save_dir = kwargs['save_dir']
        print("Saving")
        saver = kwargs['saver']
        stat_sfx = "%sit_%s_fetch.pickle" % (pfx, i)
        stats_path = os.path.join(save_dir, stat_sfx)
        pickle_it(fetch_data, stats_path)

        # Params
        params_sfx = "%sit_%s_params" % (pfx, i)
        path = os.path.join(save_dir, params_sfx)
        saver.save(sess, path)


def save_every_n(fetch_data, feed_dict, i, save_every=100, **kwargs):
    state = kwargs['state']
    if 'all_loss' in state:
        for k, v in fetch_data['loss'].items():
            state['all_loss'][k] += [v]
    else:
        state['all_loss'] = {}
        for k, v in fetch_data['loss'].items():
            state['all_loss'][k] = [v]


    if i % save_every == 0:
        save_callback(fetch_data, feed_dict, i, **kwargs)


def save_everything_last(fetch_data,
                         feed_dict,
                         i: int,
                         **kwargs):
    """Saves everything on the last iteration"""
    num_iterations = kwargs['num_iterations']
    if i == (num_iterations - 1):
        save_dir = kwargs['save_dir']
        path = os.path.join(save_dir, "state.pickle")
        pickle_it(kwargs['state'], path)
        save_callback(fetch_data, feed_dict, i, pfx='last_', **kwargs)


def save_options(fetch_data, feed_dict, i: int, **kwargs):
    """Save the options"""
    save = kwargs['save']
    if i == 0 and save:
        save_dir = kwargs['save_dir']
        options_dir = "options"
        valid_types = [list, str, float, int]
        to_save_options = {}
        for k, v in kwargs.items():
            if any((isinstance(v, typ) for typ in valid_types)):
                to_save_options[k] = v
        stats_path = os.path.join(save_dir, options_dir)
        pickle_it(to_save_options, "%s.pickle" % stats_path)
        save_it_csv(to_save_options, "%s.csv" % stats_path)
