from numpy import array, array_split
import tensorflow as tf

train_loop = 5000
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
                    [0, 0, 0, 0, 0, 1]])

train_output = array([[0], [1], [1], [1], [1], [0], [1], [0],
                      [0], [0], [1], [0], [0], [1], [0], [1],
                      [0], [1], [0], [0], [1], [0], [1], [0],
                      [1], [1], [0], [1], [1], [1], [1], [0]])

test_data = array([[1, 0, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1],
                   [0, 1, 0, 0, 1, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 0, 1, 1, 1, 0],
                   [0, 1, 0, 0, 0, 0]])

test_output = array([[0], [1], [1], [1], [1],
                     [1], [1], [0], [0], [1]])

x = tf.placeholder(tf.float32, (1, 6), name='x')

with tf.name_scope(name='later1'):
    w1 = tf.Variable(tf.random_uniform((6, 12), -1, 1), name='weight1')
    b1 = tf.Variable(tf.random_uniform((1, 12), -1, 1), name='biases1')
    y1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1, name='output1')

with tf.name_scope(name='later2'):
    w2 = tf.Variable(tf.random_uniform((12, 12), -1, 1), name='weight2')
    b2 = tf.Variable(tf.random_uniform((1, 12), -1, 1), name='biases2')
    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2, name='output2')

with tf.name_scope(name='later3'):
    w3 = tf.Variable(tf.random_uniform((12, 1), -1, 1), name='weight3')
    b3 = tf.Variable(tf.random_uniform((1, 1), -1, 1), name='biases3')
    y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3, name='output3')

label = tf.placeholder(tf.float32, (1, 1), name="label")
loss = tf.square(y3 - label, name="loss")
train_method = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs3', graph=tf.get_default_graph())
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
        print(sess.run(y3, {x: test[i]}))
