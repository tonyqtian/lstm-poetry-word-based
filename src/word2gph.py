#!/usr/bin/python3
'''
Created on Jan 13, 2017

@author: tonyq
'''

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
print(matplotlib.get_backend())
from matplotlib import pylab
from matplotlib import font_manager
from sklearn.manifold import TSNE

import argparse
import os
import json
from collections import namedtuple

from model import Model

WORK_DIR = ''
num_points = 0
OUTPUT = ''
OS = ''

def plot(embeddings, labels):
    if OS == 'win':
        myfont = font_manager.FontProperties(fname='c:\Windows\Fonts\simsun.ttc')
    else:
        myfont = font_manager.FontProperties(fname='/usr/share/fonts/truetype/moe/MoeStandardSong.ttf')
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom',fontproperties=myfont)
    if OS == 'win':
        pylab.show()
    else:
        pylab.savefig(OUTPUT)
  
def main(_):

    with open(os.path.join(WORK_DIR, 'vocab.npy'), 'rb') as fh:
        id2word = np.load(fh).tolist()
#     word2id = dict(zip(id2word, range(len(id2word))))

    with open(os.path.join(WORK_DIR, 'config.json'), 'r') as fh:
        d = json.load(fh)
    d['batch_size'] = 1
    d['num_steps'] = 1
    config = namedtuple('ModelConfig', d.keys())(*d.values())

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(is_training=False, config=config)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(WORK_DIR)
        saver.restore(session, ckpt.model_checkpoint_path)

        final_embeddings = session.run([m.embedding])
        print(len(final_embeddings[0]))
        print(len(final_embeddings[0][1:num_points+1]))

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings = tsne.fit_transform(final_embeddings[0][1:num_points+1])
        words = [id2word[i] for i in range(1, num_points+1)]
        plot(two_d_embeddings, words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM word vector viewer')
    parser.add_argument("--source", default='../data-test', help="source folder",)
    parser.add_argument("--output", default='wordvec.png', help="output file",)
    parser.add_argument("--points", default='400', help="number of points to draw",)
    parser.add_argument("--os", default='win', help="draw in windows or to file",)
    args = parser.parse_args()
    WORK_DIR = args.source
    num_points = int(args.points)
    OUTPUT = os.path.join(WORK_DIR, args.output)
    OS = args.os
    
    tf.app.run()