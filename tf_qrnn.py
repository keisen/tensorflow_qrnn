import tensorflow as tf


class QRNN():

    def __init__(self, in_size, size, conv_size=2):
        self.kernel = None
        self.batch_size = -1
        self.conv_size = conv_size
        self.c = None
        self.h = None
        self._x = None
        if conv_size == 1:
            self.kernel = QRNNLinear(in_size, size)
        elif conv_size == 2:
            self.kernel = QRNNWithPrevious(in_size, size)
        else:
            self.kernel = QRNNConvolution(in_size, size, conv_size)
    
    def initialize(self, batch_size):
        self.batch_size = batch_size
        with tf.variable_scope("QRNN"):
            self.c = tf.get_variable("c", [self.batch_size, self.kernel.size], initializer=tf.constant_initializer(0))
        self.kernel.initialize(self.batch_size)

    def _step(self, f, z, o):
        # f,z,o is batch_size x size
        if self.c is None:
            raise Exception("You have to initialize QRNN by batch size.")
        f = tf.sigmoid(f)
        z = tf.tanh(z)
        o = tf.sigmoid(o)
        self.c = tf.mul(f, self.c) + tf.mul(1 - f, z)
        self.h = tf.mul(o, self.c)  # h is size vector

        return self.h

    def forward(self, x):

        length = lambda mx: int(mx.get_shape()[0])

        if self.conv_size <= 2:
            # x is batch_size x sentence_length x word_length
            # -> now, transpose it to sentence_length x batch_size x word_length
            _x = tf.transpose(x, [1, 0, 2])

            for i in range(length(_x)):
                t = _x[i] # t is batch_size x word_length matrix
                f, z, o = self.kernel.forward(t)
                self._step(f, z, o)
        else:
            c_f, c_z, c_o = self.kernel.conv(x)
            for i in range(length(c_f)):
                f, z, o = c_f[i], c_z[i], c_o[i]
                self._step(f, z, o)
        
        return self.h


class QRNNLinear():

    def __init__(self, in_size, size):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        with tf.variable_scope("QRNNLinear"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)
    
    def initialize(self, batch_size):
        pass

    def forward(self, t):
        # x is batch_size x word_length matrix
        _weighted = tf.matmul(t, self.W)
        _weighted = tf.add(_weighted, self.b)
        
        # now, _weighted is batch_size x weight_size
        f, z, o = tf.split(1, 3, _weighted)  # split to f, z, o. each matrix is batch_size x size
        return f, z, o


class QRNNWithPrevious():

    def __init__(self, in_size, size):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        with tf.variable_scope("QRNNWithPrevious"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.V = tf.get_variable("V", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)
    
    def initialize(self, batch_size):
        with tf.variable_scope("QRNNWithPrevious"):
            self._previous = tf.get_variable("previous", [batch_size, self.in_size], initializer=tf.random_normal_initializer())

    def forward(self, t):
        _current = tf.matmul(t, self.W)
        _previous = tf.matmul(self._previous, self.V)
        _previous = tf.add(_previous, self.b)
        _weighted = tf.add(_current, _previous)

        f, z, o = tf.split(1, 3, _weighted)  # split to f, z, o. each matrix is batch_size x size
        self._previous = t
        return f, z, o


class QRNNConvolution():

    def __init__(self, in_size, size, conv_size):
        self.in_size = in_size
        self.size = size
        self.conv_size = conv_size
        self._weight_size = self.size * 3  # z, f, o

        with tf.variable_scope("QRNNConvolution"):
            initializer = tf.random_normal_initializer()
            self.conv_filter = tf.get_variable("conv_filter", [conv_size, in_size, self._weight_size], initializer=initializer)
    
    def initialize(self, batch_size):
        pass

    def conv(self, x):
        # !! x is batch_size x sentence_length x word_length(=channel) !!
        _weighted = tf.nn.conv1d(x, self.conv_filter, stride=1, padding="SAME", data_format="NHWC")

        # _weighted is batch_size x conved_size x output_channel
        _w = tf.transpose(_weighted, [1, 0, 2])  # conved_size x  batch_size x output_channel
        _ws = tf.split(2, 3, _w) # make 3(f, z, o) conved_size x  batch_size x size
        return _ws
