import tensorflow as tf
import numpy as np

def batch_normalization(x):
    axis = range(len(x.get_shape()) - 1)
    m, v = tf.nn.moments(x, axis)
    h = tf.nn.batch_normalization(x, m, v, None, None, 1e-7)
    return h

def linear(name, x, out_dim):
    with tf.variable_scope("Liner/{}".format(name)):
        w = tf.get_variable("weights",
                            shape=[ x.get_shape()[1], out_dim ],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True))
        b = tf.get_variable("biases",
                            shape=[ out_dim ],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
    h = tf.matmul(x, w)
    h = tf.nn.bias_add(h, b)
    return h

def conv2d(name, x, filter_shape, strides=[ 1, 1, 1, 1 ]):
    with tf.variable_scope("CNN/{}".format(name)):
        w = tf.get_variable("weights",
                            shape=filter_shape,
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True))
        b = tf.get_variable("biases",
                            shape=filter_shape[-1],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
    h = tf.nn.conv2d(x, w, strides, 'SAME')
    h = tf.nn.bias_add(h, b)
    return h

def qrnn_conv2d(name, x, filter_shape, batch_size, strides=[ 1, 1, 1, 1 ], dap='ifo'):
    if dap is 'ifo':
        channel_times = 4
    elif dap is 'fo':
        channel_times = 3
    elif dap is 'f':
        channel_times = 2
    else:
        return conv2d(name, x, filter_shape, strides)
    if dap in [ 'ifo', 'fo' ]:
        with tf.variable_scope("QRNN/{}".format(name)):
            c = tf.get_variable("context",
                                shape=[ x.get_shape()[2], filter_shape[-1] ],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
    filter_shape = (filter_shape * np.asarray([ 1, 1, 1, channel_times ])).tolist()
    h = conv2d(name, x, filter_shape, strides)
    h = tf.transpose(h, [ 1, 0, 2, 3 ])
    if dap is 'ifo':
        z, f, o, i = tf.split(h, channel_times, 3)
        h = _ifo_pooling(z, f, o, i, c, batch_size)
    elif dap is 'fo':
        z, f, o = tf.split(h, channel_times, 3)
        h = _fo_pooling(z, f, o, c, batch_size)
    else:
        z, f = tf.split(h, channel_times, 3)
        h = _f_pooling(z, f)
    h = tf.transpose(h, [ 1, 0, 2, 3 ])
    return h

def _f_pooling(z, f):
    z = tf.tanh(z)
    f = tf.sigmoid(f)
    h = []
    for n in xrange(z.get_shape()[0]):
        if len(h) > 0:
            o = tf.multiply(f[n], h[n-1]) + tf.multiply(1 - f[n], z[n])
        else:
            o = tf.multiply(1 - f[n], z[n])
        h.append(o)
    h_shape = map(lambda a: -1 if a is None else a ,z.get_shape().as_list())
    return tf.reshape(tf.concat(h, 0), h_shape)
    
def _fo_pooling(z, f, o, context, batch_size):
    z = tf.tanh(z)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    c = tf.concat([[ context.value() ]] * batch_size, 0)
    h = []
    for n in xrange(z.get_shape()[0]):
        c = tf.multiply(f[n], c) + tf.multiply(1 - f[n], z[n])
        h.append(tf.multiply(o[n], c))
    tf.assign(context, tf.reduce_mean(c, 0))
    h_shape = map(lambda a: -1 if a is None else a ,z.get_shape().as_list())
    return tf.reshape(tf.concat(h, 0), h_shape)

def _ifo_pooling(z, f, o, i, context, batch_size):
    z = tf.tanh(z)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    i = tf.sigmoid(i)
    c = tf.concat([[ context.value() ]] * batch_size, 0)
    h = []
    for n in xrange(z.get_shape()[0]):
        c = tf.multiply(f[n], c) + tf.multiply(i[n], z[n])
        h.append(tf.multiply(o[n], c))
    tf.assign(context, tf.reduce_mean(c, 0))
    h_shape = map(lambda a: -1 if a is None else a ,z.get_shape().as_list())
    return tf.reshape(tf.concat(h, 0), h_shape)

