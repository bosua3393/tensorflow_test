import tensorflow as tf
import numpy as np

train_data = np.array([[1, 0, 1, 1, 0, 0],
                       [1, 1, 1, 0, 1, 0],
                       [1, 1, 1, 0, 0, 1],
                       [1, 0, 0, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 0, 1],
                       [0, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 0],
                       [1, 0, 1, 0, 0, 1],
                       [1, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 0, 1],
                       [0, 0, 1, 0, 1, 1]])

train_output = np.array([[1], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [0]])


b = tf.Variable(tf.zeros(1))
w = tf.Variable(tf.random_uniform((6, 1), -1, 1))

x = tf.placeholder(tf.float32, (12, 6))

y = tf.nn.sigmoid(tf.matmul(x, w) + b)

session = tf.Session()
session.run(tf.global_variables_initializer())


prediction = session.run(y, {x: train_data})
label = tf.placeholder(tf.float32, [12, 1])
loss = label - prediction

train_step = tf.train.GradientDescentOptimizer(.01).minimize(loss)


# https://www.youtube.com/watch?v=PicxU81owCs