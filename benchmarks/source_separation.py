"""Source Separation"""
from arrows.config import floatX
import glob
import os
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from functools import reduce
from arrows.util.misc import getn, pull
import sys
from common import handle_options, gen_sfx_key
from reverseflow.invert import invert
from reverseflow.to_arrow import graph_to_arrow
from reverseflow.train.train_y import min_approx_error_arrow
from reverseflow.train.loss import inv_fwd_loss_arrow, supervised_loss_arrow
from reverseflow.train.common import get_tf_num_params
from arrows.port_attributes import *
from arrows.apply.propagate import *
from reverseflow.train.reparam import *
from reverseflow.train.unparam import unparam
from reverseflow.train.callbacks import save_callback, save_options, save_every_n, save_everything_last
from reverseflow.train.supervised import supervised_train
from arrows.util.misc import flat_dict


# Get Data
datadir = os.environ['DATADIR']
audio_dir = os.path.join(datadir, "UrbanSound8K", "audio")
fold1_dir = os.path.join(audio_dir, "fold1")
sound_file_paths = ["57320-0-0-7.wav","24074-1-0-3.wav","15564-2-0-1.wav","31323-3-0-1.wav","46669-4-0-35.wav",
                   "89948-5-0-0.wav","40722-8-0-4.wav","103074-7-3-2.wav","106905-8-0-0.wav","108041-9-0-4.wav"]
fold1_paths = sorted(glob.glob("%s/*" % fold1_dir))
# full_sound_paths = [os.path.join(fold1_dir, wav) for wav in sound_file_paths]
# # raw_sounds, srs = load_sound_files(full_sound_paths)
# raw_sounds, srs = load_sound_files(fold1_paths)
# sound_batch = np.array(raw_sounds)
fold1_dataset = np.load(os.path.join(fold1_dir, "fold1.npz")).items()[0][1]
sound_len = fold1_dataset[0].shape[0]


## Sound manipulation and stuff
def load_sound_files(file_paths):
    raw_sounds = []
    srs = []
    for fp in file_paths:
        X, sr = librosa.load(fp, duration=1.0)
        raw_sounds.append(X)
        srs.append(sr)
    return raw_sounds, srs


def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()


def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

## Tensorflow stuff
def tf_inputs(n_sources, batch_size, source_len, ndim):
    sources = []
    positions = []
    for i in range(n_sources):
        source = tf.placeholder(name="source", shape=(batch_size, source_len), dtype=floatX())
        sources.append(source)
        position = tf.placeholder(name="pos", shape=(batch_size, ndim), dtype=floatX())
        positions.append(position)
    return sources, positions

## Tensorflow stuff
def tf_dist_inputs(n_sources, batch_size, source_len):
    sources = []
    distances = []
    for i in range(n_sources):
        source = tf.placeholder(name="source", shape=(batch_size, source_len), dtype=floatX())
        sources.append(source)
        distance = tf.placeholder(name="dist", shape=(batch_size, 1), dtype=floatX())
        distances.append(distance)
    return sources, distances


def euclidean_dist(a, b):
    return tf.reduce_sum(tf.square(b - a), reduction_indices=1)


def weight(a, b):
    return euclidean_dist(a, b)


def reduce_sum(xs):
    return reduce((lambda x, y: x + y), xs)


def mixing_model(n_sources, batch_size=None, source_len=None, ndim=3, pos=False):
    """Create the mixing mode
    """
    receiver_pos = np.array([[0.0,0.0,0.0]])
    if pos:
        sources, positions = tf_inputs(n_sources=n_sources, batch_size=batch_size,
                                       source_len=source_len, ndim=ndim)
        weights = [weight(receiver_pos, pos) for pos in positions]
        source_pos_dist = positions
    else:
        sources, distances = tf_dist_inputs(n_sources=n_sources, batch_size=batch_size,
                                      source_len=source_len)
        weights = distances
        source_pos_dist = distances

    weighted_signal = [weights[i] * sources[i] for i in range(len(sources))]
    mix_signal = reduce_sum(weighted_signal)
    return {'sources': sources,
            'source_pos_dist': source_pos_dist,
            'weights': weights,
            'weighted_signal': weighted_signal,
            'mix_signal': mix_signal}


def mixing_model_tf(batch_size, **options):
    model = mixing_model(n_sources=3, batch_size=batch_size, source_len=sound_len)
    sources, source_pos_dist = getn(model, 'sources', 'source_pos_dist')
    inputs = sources + source_pos_dist
    outputs = [model['mix_signal']]
    return {'inputs': inputs, 'outputs': outputs}


def example(model):
    sess = tf.InteractiveSession()
    sources, positions = getn(model, 'sources', 'positions')
    inputs1 = [[0.5, 0.5, 0.5]]
    inputs2 = [[1.0, 0.7, 0.3]]
    inputs3 = [[0.4, -1.0, 2.0]]
    inputs = [inputs1, inputs2, inputs3]
    source1 = np.expand_dims(sound_batch[0],axis=0)
    source2 = np.expand_dims(sound_batch[1],axis=0)
    source3 = np.expand_dims(sound_batch[2],axis=0)
    data_sources = [source1, source2, source3]
    pos_feed = {positions[i]: inputs[i] for i in range(len(positions))}
    source_feed = {sources[i]: data_sources[i] for i in range(len(sources))}
    feed_dict = {}
    feed_dict.update(pos_feed)
    feed_dict.update(source_feed)
    re = sess.run(pull(model, 'mix_signal', 'sources', 'weighted_signal'), feed_dict=feed_dict)
    sess.close()
    return re

def write_wavs(fetch, path):
    ch = flat_dict(fetch)
    def fix_shape(x):
        return x.reshape(-1)
    for k, v in ch.items():
        fname = os.path.join(path, "%s.wav" % k)
        librosa.output.write_wav(fname, fix_shape(v), sr=22050)

sound_path = "/home/zenna/data/rf/sounds"
# fetch_data = example(model)
# write_wavs(fetch_data, path=sound_path)

def fwd_arrow(batch_size):
    model = mixing_model(3, batch_size=batch_size, source_len=sound_len)
    sources, source_pos_dist = getn(model, 'sources', 'source_pos_dist')
    inputs = sources + source_pos_dist
    outputs = [model['mix_signal']]
    arrow = graph_to_arrow(outputs,
                           input_tensors=inputs,
                           name="source_separation")
    return arrow


def gen_sound_data(batch_size, model_tensorflow, options):
    """Generate data for training"""
    graph = tf.Graph()
    # n = fold1_dataset.shape[0]
    ns = batch_size
    fold1_a = fold1_dataset[0:ns]
    fold1_b = fold1_dataset[ns:ns+ns]
    fold1_c = fold1_dataset[ns+ns:ns+ns+ns]
    source_pos_dist_data = [np.random.rand(batch_size, 1) for i in range(3)]
    input_data = [fold1_a, fold1_b, fold1_c] + source_pos_dist_data

    with graph.as_default():
        model = mixing_model(n_sources=3, batch_size=batch_size, source_len=sound_len)
        sources, source_pos_dist = getn(model, 'sources', 'source_pos_dist')
        inputs = sources + source_pos_dist
        outputs = [model['mix_signal']]
        sess = tf.Session()
        output_data = sess.run(outputs, feed_dict=dict(zip(inputs, input_data)))
        sess.close()
    return {'inputs': input_data, 'outputs': output_data}

from common import pi_benchmarks
if __name__ == "__main__":
    options = {'model': mixing_model_tf,
               'n_inputs': 3,
               'n_outputs' : 2,
               'gen_data': gen_sound_data,
               'model_name': 'source_separation',
               'error': ['supervised_error']}
    pi_benchmarks('source_separation', options=options)
