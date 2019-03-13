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

test_data = np.array([[0, 0, 1, 0, 1, 1]])

test_output = np.array([[1], [1], [0], [1], [0]])

b = tf.Variable(tf.zeros(1))
w = tf.Variable(tf.random_uniform((6, 1), -1, 1))

x = tf.placeholder(tf.float32, (1, 6))

y = tf.nn.sigmoid(tf.matmul(x, w) + b)

label = tf.placeholder(tf.float32, [1, 1])

loss = (y - label) * (y - label)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(20).minimize(loss)

for i in range(1000):
    batch_x = np.array_split(train_data, len(train_data))
    batch_label = np.array_split(train_output, len(train_output))
    loss_value = 0.0
    for batch in range(len(batch_x)):
        sess.run(train_step, feed_dict={x: batch_x[batch], label: batch_label[batch]})
        if i % 100 == 0:
            loss_value += sess.run(loss, feed_dict={x: batch_x[batch], label: batch_label[batch]})
    if i % 100 == 0:
        print(loss_value/len(batch_x))

prediction = sess.run(y, {x: test_data})

print(prediction)
