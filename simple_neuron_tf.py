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

test_data = np.array([[1, 0, 1, 0, 1, 1],
                      [1, 0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 1, 0],
                      [1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 1, 1]])

test_output = np.array([[1], [1], [0], [1], [0]])

b = tf.Variable(tf.zeros(1))
w = tf.Variable(tf.random_uniform((6, 1), -1, 1))

x = tf.placeholder(tf.float32, (12, 6))

y = tf.nn.sigmoid(tf.matmul(x, w) + b)

label = tf.placeholder(tf.float32, [12, 1])

loss = (y - label) * (y - label)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(.01).minimize(loss)

for i in range(10000):
    sess.run(train_step, feed_dict={x: train_data, label: train_output})

prediction = sess.run(y, {x: train_data})

print(np.round(prediction))
