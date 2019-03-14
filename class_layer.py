import tensorflow as tf


class layer:
    def __init__(self, x, connection, neuron, name, last_layer=False):
        with tf.name_scope(name=name):
            self.w = tf.Variable(tf.random_normal((connection, neuron), -1, 1), name="weights")
            self.b = tf.Variable(tf.random_normal((1, neuron), -1, 1), name="bias")
            self.y = tf.nn.sigmoid(tf.matmul(x, self.w) + self.b, name="output")


