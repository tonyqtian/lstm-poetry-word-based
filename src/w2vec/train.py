#!/usr/bin/python3

from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os

import argparse
from collections import namedtuple
import json
import glob

import matplotlib
matplotlib.use('Agg')
print(matplotlib.get_backend())
import matplotlib.pyplot as plt

import model
import reader

'''
    Small config:
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

    Medium config:
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

    Large config:
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
'''

WORK_DIR = '../../data-text8'

nn_config = {
    'init_scale': 0.1,
    'max_grad_norm': 5,
    'num_layers': 2,
    'num_steps': 15,
    'hidden_size': 40,
    'keep_prob': 0.6,
    'batch_size': 16,
    'vocab_size': 400
}

train_config = {
    'max_max_epoch': 15,
    'max_epoch': 10,
    'learning_rate': 1.0,
    'lr_decay': 0.6
}

num_steps = 10001

def train_w3v(session, m, data, num_steps, reverse_dictionary, verbose=False):
    print('Training Word Embedding...')
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = reader.generate_batch(data, 
          m.batch_size, m.num_skips, m.skip_window)
        feed_dict = {m.train_dataset : batch_data, m.train_labels : batch_labels}
        _, l = session.run([m.optimizer, m.loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = m.similarity.eval()
            for i in range(m.valid_size):
                valid_word = reverse_dictionary[m.valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    return m.normalized_embeddings.eval()

def run_epoch(session, m, data, eval_op, verbose=False):
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = m.initial_state.eval()
    state = session.run(m.initial_state)
    for step, (x, y) in enumerate(reader.train_iterator(data, m.batch_size,
                                                      m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps
        perplexity = np.exp(costs / iters)

        if verbose and step % (epoch_size // 10) == 10:
            print("%.2f %% perplexity: %.3f speed: %.0f wps" %
                  (step * 100.0 / epoch_size, perplexity,
                   iters * m.batch_size / (time.time() - start_time)))
                
    return perplexity


def main():

    # cleanup input dir
    ret = input('Are you sure you want to clean %s [yes|no] ' % (WORK_DIR,))
#     ret = 'yes'
    if ret == 'yes':
        for f in glob.glob(os.path.join(WORK_DIR, '*')):
            if not f.endswith('.txt'):
                os.remove(f)
                print(f + ' deleted')

    config = namedtuple('TrainConfig', train_config.keys())(*train_config.values())
    model_config = namedtuple('ModelConfig', nn_config.keys())(*nn_config.values())

    with open(os.path.join(WORK_DIR, 'config.json'), 'w') as fh:
        json.dump(nn_config, fh)

    proc = reader.TextProcessor.from_file(os.path.join(WORK_DIR, 'input.txt'))
    proc.create_vocab(model_config.vocab_size)
    train_data = proc.get_vector()  # [11 22 34 456 124 852 412 45 325 ... ]
    np.save(os.path.join(WORK_DIR, 'vocab.npy'), np.array(proc.id2word))
    proc.save_converted(os.path.join(WORK_DIR, 'input.conv.txt'))

    perplexity_graph = []
    iter_graph = []
        
    with tf.Graph().as_default(), tf.Session() as session:
#         logwriter = tf.summary.FileWriter(WORK_DIR, graph=tf.get_default_graph())
        with tf.variable_scope('model', reuse=None):
            w2v = model.Word2Vec(model_config.vocab_size, model_config.hidden_size)
        final_embeddings = train_w3v(session, w2v, train_data, num_steps, 
                                     proc.id2word, verbose=True)
#         print(final_embeddings)
            
        initializer = tf.random_uniform_initializer(-model_config.init_scale,
                                                    model_config.init_scale)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = model.Model(is_training=True, config=model_config, embd=final_embeddings)
        
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
#         print(m.embedding.eval())
#         raise(RuntimeError)
        
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                        verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            perplexity_graph.append(train_perplexity)
            iter_graph.append(i)

            ckp_path = os.path.join(WORK_DIR, 'model.ckpt')
            saver.save(session, ckp_path, global_step=i)
            
    plt.plot(iter_graph, perplexity_graph)
    plt.xlabel('iterations')
    plt.ylabel('perplexity')
#     plt.show()
    plt.savefig(os.path.join(WORK_DIR, 'learning_curve.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM trainer')
    parser.add_argument("--source", help="source folder",)
    parser.add_argument("--num_layers", type=int, help="LSTM layers",)
    parser.add_argument("--hidden_size", type=int, help="LSTM hidden layers size",)
    parser.add_argument("--keep_prob", type=float, help="LSTM dropout",)
    parser.add_argument("--vocab_size", type=int, help="Input data vocab max size",)
    parser.add_argument("--batch_size", type=int, help="Batch size",)
    parser.add_argument("--num_steps", type=int, help="length of every training piece",)
    parser.add_argument("--epoch", type=int, help="Max epoch",)
        
    args = parser.parse_args()
    if args.source:
        WORK_DIR = args.source
    if args.num_layers:
        nn_config['num_layers'] = args.num_layers
    if args.hidden_size:
        nn_config['hidden_size'] = args.hidden_size
    if args.keep_prob:
        nn_config['keep_prob'] = args.keep_prob
    if args.vocab_size:
        nn_config['vocab_size'] = args.vocab_size
    if args.num_steps:
        nn_config['num_steps'] = args.num_steps
    if args.batch_size:
        nn_config['batch_size'] = args.batch_size
    if args.epoch:
        train_config['max_max_epoch'] = args.epoch
        train_config['max_epoch'] = int(args.epoch * 0.6)
    main()
