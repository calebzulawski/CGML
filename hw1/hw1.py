#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 50
xs = np.linspace(-np.pi, np.pi, N, dtype=np.float64)
ys = np.sin(xs) + np.random.normal(0, 0.1, N)
xplot = np.linspace(-4, 4, 1000)

M = 20

two = tf.constant(2, dtype=tf.float64)
x = tf.placeholder(tf.float64) # input
y = tf.placeholder(tf.float64) # output
mu = tf.Variable(tf.random_normal([M, 1], dtype=tf.float64), dtype=tf.float64, name="mean")
var = tf.Variable(tf.ones([M, 1], dtype=tf.float64), dtype=tf.float64, name="variance")
w =  tf.Variable(tf.random_normal([1, M], dtype=tf.float64), dtype=tf.float64, name="weight")
b = tf.Variable(0, dtype=tf.float64, name="bias")

gaussian = tf.exp(tf.negative(tf.divide(tf.square(tf.sub(x, mu)), var)))


yest = tf.add(tf.matmul(w, gaussian), b)

loss = tf.reduce_sum(tf.divide(tf.squared_difference(y, yest), two))

learning_rate = 0.01
training_epochs = 1000
display_step = 50

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (xi, yi) in zip(xs, ys):
            sess.run(optimizer, feed_dict={x: xi, y: yi})

        if (epoch+1) % display_step == 0:
            print(sess.run(loss, feed_dict={x: xs, y: ys}))

    print(sess.run(yest, feed_dict={x: xs}))

    plt.plot(xs, ys, 'ro', label='Original data')
    plt.plot(xplot, sess.run(yest, feed_dict={x: xplot})[0], 'b', label='Fitted line')
    plt.legend()
    plt.show()

    plt.plot(xplot, sess.run(tf.transpose(gaussian+b), feed_dict={x: xplot}))
    plt.show()
