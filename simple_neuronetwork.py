from numpy import array, array_split
import tensorflow as tf
from class_layer import layer

train_loop = 5000
learn_rate = 2

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

train_output = array([[1], [1], [1], [0], [0], [0], [0], [0],
                      [0], [0], [1], [1], [1], [0], [0], [1],
                      [1], [1], [1], [1], [0], [0], [1], [1],
                      [0], [1], [0], [1], [0], [0], [0], [0]])

test_data = array([[1, 0, 1, 0, 1, 1],
                   [1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1],
                   [0, 1, 0, 0, 1, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 0, 1, 1, 1, 0],
                   [0, 1, 0, 0, 0, 0]])

test_output = array([[0], [1], [0], [0], [0],
                     [1], [1], [0], [1], [1]])

x = tf.placeholder(tf.float32, (1, 6), name='x')

layer1 = layer(x, 6, 36, "layer1")
layer2 = layer(layer1.y, 36, 12, "layer2")
layer3 = layer(layer2.y, 12, 6, "layer3")
layer4 = layer(layer3.y, 6, 1, "layer4")

label = tf.placeholder(tf.float32, (1, 1), name="label")
loss = tf.losses.mean_squared_error(label, layer4.y)
train_method = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('./graphs3', graph=tf.get_default_graph())
    for step in range(train_loop):
        batch_x = array_split(train_data, len(train_data))
        batch_label = array_split(train_output, len(train_output))
        loss_value = 0.0
        for i in range(len(train_data)):
            sess.run(train_method, {x: batch_x[i], label: batch_label[i]})
            if step % 200 == 0:
                loss_value += sess.run(loss, {x: batch_x[i], label: batch_label[i]})
                if i == 31:
                    print(loss_value)

    print('testing')
    test = array_split(test_data, len(test_data))
    for i in range(len(test_data)):
        print(sess.run(layer4.y, {x: test[i]}))
