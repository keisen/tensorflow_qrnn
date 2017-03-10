from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import sys
import random
import time

import numpy as np
import tensorflow as tf

import keisen as ks
import dataset as ds

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=100, type=int,
                        help='batch_size')
    parser.add_argument('--epoch',
                        default=50, type=int,
                        help='epoch')
    return parser.parse_args()

def main(args):
    start = time.time()
    dataset = _load_datasets()
    print("Dataset loaded ({}sec)".format(time.time() - start))

    x = tf.placeholder(tf.float32,
                       shape=[ None,
                               dataset.unit_length(),
                               dataset.columns_size(),
                               dataset.channel_size() ],
                       name="input")
    t = tf.placeholder(tf.int64,
                       shape=[ None ],
                       name="label")

    #graph    = _graph(args, x, dataset.channel(), dataset.label_size())
    graph = _graph(args, x, dataset.channel_size(), 2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=graph, labels=t)
    loss = tf.reduce_mean(cross_entropy)
    train = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
    correct = tf.equal(tf.argmax(graph, 1), t)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #loss     = tf.reduce_sum(tf.squared_difference(graph, t))
    #accuracy = tf.sqrt(tf.reduce_mean(tf.squared_difference(graph, t)))
    #accuracy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=graph, labels=t))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('temp/logs', graph=sess.graph)

        for epoch in xrange(1, args.epoch + 1):
            dataset.reset()
            train_loss     = 0
            train_accuracy = 0
            train_step     = 0
            test_loss      = 0
            test_accuracy  = 0
            test_step     = 0
            start = time.time()
            for step in xrange(1, int(1e8)):
                x1, t1 = _batch(args, dataset.train)
                if x1 is None:
                    break
                x1 = np.reshape(x1, [ -1, dataset.unit_length(),
                    dataset.columns_size(), dataset.channel_size() ])
                sess.run(train, feed_dict={ x: x1, t: t1 })
                train_loss = sess.run(loss, feed_dict={ x: x1, t: t1 })
                train_accuracy = sess.run(accuracy, feed_dict={ x: x1, t: t1 })
            for step in xrange(1, int(1e8)):
                x1, t1 = _batch(args, dataset.test)
                if x1 is None:
                    break
                x1 = np.reshape(x1, [ -1, dataset.unit_length(),
                    dataset.columns_size(), dataset.channel_size() ])
                test_loss = sess.run(loss, feed_dict={ x: x1, t: t1 })
                test_accuracy = sess.run(accuracy, feed_dict={ x: x1, t: t1 })
            print("epoch: {},\ttrain_loss: {},\ttest_loss: {},\ttrain_accuracy: {},\ttest_accuracy: {}, time: {}sec".
                    format(epoch,
                           train_loss,
                           test_loss,
                           train_accuracy,
                           test_accuracy,
                           time.time() - start))

def _batch(args, dataset):
    x = None
    t = None
    for i in xrange(args.batch_size):
        if x is None:
            x, t = dataset.get()
            if x is None:
                return None, None
            else:
                t = [ t ]
        else:
            x1, t1 = dataset.get()
            if x1 is None:
                return None, None
            x = np.concatenate((x, x1), 0)
            t.append(t1)
    return x, np.asarray(t)

def _load_datasets():
    return ds.load('data/index', 'data/stock')

def _graph(args, x, in_channel, out_size):
    """
    h = ks.conv2d("conv1", x, [ 3, 3, in_channel, 24 ])
    h = ks.batch_normalization(h)
    h = tf.nn.relu(h)
    h = ks.qrnn_conv2d("conv2", h, [ 3, 3, 24, 24 ], args.batch_size, dap='f')
    """
    h = ks.qrnn_conv2d("conv1", x, [ 3, 3, in_channel, 12 ], args.batch_size, dap='f')

    #h = ks.qrnn_conv2d("conv2", h, [ 3, 3, 24, 32 ], args.batch_size, dap='ifo')
    #h = ks.batch_normalization(h)
    #h = tf.nn.relu(h)
    """
    h = tf.nn.max_pool(h,
                       ksize  =[ 1, 2, 2, 1 ],
                       strides=[ 1, 2, 2, 1 ],
                       padding="VALID")
    #h = ks.conv2d("conv2", h, [ 3, 3, 24, 24 ])
    h = ks.qrnn_conv2d("conv2", h, [ 3, 3, 24, 32 ], args.batch_size, dap='f')
    h = ks.batch_normalization(h)
    h = tf.nn.relu(h)
    h = tf.nn.max_pool(h,
                       ksize  =[ 1, 2, 2, 1 ],
                       strides=[ 1, 2, 2, 1 ],
                       padding="VALID")
    """

    #h = concat([ h ])
    h = concat([ tf.transpose(h, [ 1, 0, 2, 3 ])[-1] ])
    #h = ks.batch_normalization(h)
    h = tf.nn.relu(h)

    #h = tf.nn.dropout(h, 0.5)

    """
    h  = _fc("fc4", h, 64)
    h  = _fc("fc5", h, 32)
    """
    o  = _out("out", h, out_size)

    return o

def _conv(name, x, filter, batch_size, pooling=True):
    h, p = ks.qrnn_conv2d(name, x,
                          filter=filter,
                          strides=[ 1, 1, 1, 1 ],
                          batch_size=batch_size)
    h = ks.batch_normalization(h)
    h = tf.nn.relu(h)
    if pooling:
        h = tf.nn.max_pool(h,
                           ksize  =[ 1, 2, 2, 1 ],
                           strides=[ 1, 2, 2, 1 ],
                           padding="VALID")
    p = ks.batch_normalization(p)
    if pooling:
        p = tf.nn.max_pool(p,
                           ksize  =[ 1, 2, 2, 1 ],
                           strides=[ 1, 2, 2, 1 ],
                           padding="VALID")
    p = tf.nn.relu(p)
    return h, p

def concat(features):
    def dimention(x): 
        shape = x.get_shape()
        dim   = 1
        for i in xrange(1, len(shape)):
            dim *= shape[i].value
        return dim
    features = map(lambda x: tf.reshape(x, [ -1, dimention(x) ]), features)
    feature  = tf.concat(features, 1)
    print("Fc-layer input size is {}.".format(feature.shape))
    return feature

def _fc(name, x, out_dim):
    h = ks.linear(name, x, out_dim)
    h = ks.batch_normalization(h)
    h = tf.nn.relu(h)
    return h

def _out(name, x, out_dim):
    h = ks.linear(name, x, out_dim)
    h = tf.nn.softmax(h)
    return h

if __name__ == '__main__':
    args = arguments()
    main(args)

