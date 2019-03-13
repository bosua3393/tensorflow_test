import tensorflow as tf
from numpy import array, array_split, round
import time

init_time = int(round(time.time() * 1000))


train_loop = 2000
learn_rate = 5

train_data = array([[1, 0, 1, 1, 0, 0],
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
                    [0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1],

                    [1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 1],

                    [1, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    ])

train_output = array([[1], [0], [0], [1], [1], [1], [1], [1],
                      [1], [0], [1], [1], [1], [1], [0], [1],
                      [1], [1], [1], [1], [0], [0], [1], [1],
                      [1], [0], [0], [1], [0], [0], [1], [0]])

test_data = array([[1, 0, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0, 0],
                   [1, 1, 0, 1, 1, 0],
                   [0, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 1]])

test_output = array([[1], [1], [0], [0], [1]])

x = tf.placeholder(tf.float32, (1, 6), name="x")

with tf.name_scope(name="layer1") as layer1:
    w1 = tf.Variable(tf.random.uniform((6, 6), -1, 1), name="weights1")
    b1 = tf.Variable(tf.random.uniform((1, 6), -1, 1), name="biases1")
    y1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1, name="output1")

with tf.name_scope(name="layer2") as layer2:
    w2 = tf.Variable(tf.random.uniform((6, 1), -1, 1), name="weights2")
    b2 = tf.Variable(tf.random.uniform((1, 1), -1, 1), name="bias2")
    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2, name="output2")

label = tf.placeholder(tf.float32, (1, 1), name="label")
loss = tf.square(y2 - label, name="loss")
train_method = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs2', graph=tf.get_default_graph())
    for step in range(train_loop):
        batch_x = array_split(train_data, len(train_data))
        batch_label = array_split(train_output, len(train_output))
        loss_value = 0.0
        for i in range(len(train_data)):
            sess.run(train_method, {x: batch_x[i], label: batch_label[i]})
            if step % 200 == 0:
                loss_value += sess.run(loss, {x: batch_x[i], label: batch_label[i]})
                if i == 11:
                    print(loss_value)

    print('testing')
    test = array_split(test_data, len(test_data))
    for i in range(len(test_data)):
        print(round(sess.run(y2, {x: test[i]}), 8))

time = int(round(time.time() * 1000)) - init_time
print(time)



