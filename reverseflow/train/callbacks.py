"""Generalization Test"""

# So its better to prevent train_loop from becoming this huge mess
# and instead pass in a callback
# a lens would be even better but there you go, we dont have that
import numpy as np
import os


def save_callback(fetch_data,
                  feed_dict,
                  i: int,
                  save_every=100,
                  compress=True,
                  **kwargs):
    save_dir = kwargs['save_dir']
    sess = kwargs['sess']
    save = kwargs['save']
    loss = fetch_data['loss']
    if i % save_every == 0 and save:
        print("Saving")
        saver = kwargs['saver']
        stat_sfx = "it_%s_fetch" % i
        stats_path = os.path.join(save_dir, stat_sfx)
        if compress:
            np.savez_compressed(stats_path, **fetch_data)
        else:
            np.savez(stats_path, **fetch_data)
        # Save Params
        params_sfx = "it_%s_params" % i
        path = os.path.join(save_dir, params_sfx)
        saver.save(sess, path)
