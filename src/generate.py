#!/usr/bin/python3

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import reader
import os
import sys
import re
import json
from collections import namedtuple

from model import Model

WORK_DIR = '.'
OUTPUT = ''
word_length = 0
TEMPERATURE = .7
INI_TEXT = '''ocean and what'''


def weighted_pick(a):
    a = a.astype(np.float64)
    a = a.clip(min=1e-20)
    a = np.log(a) / TEMPERATURE
    a = np.exp(a) / (np.sum(np.exp(a)))
    return np.argmax(np.random.multinomial(1, a, 1))


class Printer:
    def __init__(self):
        self.re1 = re.compile(r'^\W')
        self.prev = ''

    def print_word(self, word):
        if self.prev == '\n' or self.prev == '('\
                or word[0] == "'" or word in '.,:;)?!'\
                or (word[0] == '-' and len(word) > 1):
            s = ''
        else:
            s = ' '
        s += word
        self.prev = word
        sys.stdout.write(s)
        sys.stdout.flush()


def main(_):

    with open(os.path.join(WORK_DIR, 'vocab.npy'), 'rb') as fh:
        id2word = np.load(fh).tolist()
    word2id = dict(zip(id2word, range(len(id2word))))

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

        state = session.run(m.initial_state)

        data = reader.TextProcessor(INI_TEXT).set_vocab(word2id).get_vector()
        sys.stdout.write(INI_TEXT)

        w_id = 0
        for w_id in data:
            x = np.zeros((1, 1), dtype=np.int32)
            x[0, 0] = w_id
            logits, state = session.run([m.logits, m.final_state], {
                m.input_data: x, m.initial_state: state})

        p = Printer()
#         print(word2id['<unk>'])
#         print(id2word[word2id['<unk>']])
        for _ in range(word_length):
            x = np.zeros((1, 1), dtype=np.int32)
            x[0, 0] = w_id
            logits, state = session.run([m.logits, m.final_state], {
                m.input_data: x, m.initial_state: state})

            probs = session.run(tf.nn.softmax(logits)).flatten()
            w_id = weighted_pick(probs)
            if id2word[w_id] == '<unk>':
                pass
            else:
                p.print_word(id2word[w_id])

        sys.stdout.write('\n\n')
        print('Finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM generator')
    parser.add_argument("--source", default='../data-test', help="source folder",)
    parser.add_argument("--output", default='sample.txt', help="output filename",)
    parser.add_argument("--length", default=100, type=int, help="generate length",)
    parser.add_argument("--header", help="header for generation",)

    args = parser.parse_args()
    WORK_DIR = args.source
    OUTPUT = os.path.join(WORK_DIR, args.output)
    word_length = args.length
    if args.header:
        INI_TEXT = args.header
    
    tf.app.run()
