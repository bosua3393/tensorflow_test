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

x = tf.placeholder(tf.float32, (1, 6), name="x")

with tf.name_scope(name="layer1"):
    b = tf.Variable(tf.zeros(1), name="bias")
    w = tf.Variable(tf.random_uniform((6, 1), -1, 1), name="weights")
    y = tf.nn.sigmoid(tf.matmul(x, w) + b, name="y")

label = tf.placeholder(tf.float32, [1, 1], name="label")
loss = tf.square(y - label, name="loss")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())

train_step = tf.train.GradientDescentOptimizer(25).minimize(loss)

for i in range(1000):
    batch_x = np.array_split(train_data, len(train_data))
    batch_label = np.array_split(train_output, len(train_output))
    loss_value = 0.0

    for batch in range(len(batch_x)):
        sess.run(train_step, feed_dict={x: batch_x[batch], label: batch_label[batch]})
        if i % 100 == 0:
            loss_value += sess.run(loss, feed_dict={x: batch_x[batch], label: batch_label[batch]})

    if i % 100 == 0:
        print(loss_value / len(batch_x))


print('testing')
test = np.array_split(test_data, len(test_data))
for i in range(len(test_data)):
    print(np.round(sess.run(y, {x: test[i]}), 3))
