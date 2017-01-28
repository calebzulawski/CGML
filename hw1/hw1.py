#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 50
xs = np.linspace(-np.pi, np.pi, N, dtype=np.float32)
ys = np.sin(xs) + np.random.normal(0, 0.1, N)

class Model():
    def __init__(self, sess, n_epochs, learning_rate, M):
        self.sess = sess
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.M = M
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32) # input
        self.y = tf.placeholder(tf.float32) # output

        two = tf.constant(2, dtype=tf.float32)

        self.mu = tf.get_variable(shape=[self.M, 1], dtype=tf.float32, name="mean", initializer=tf.random_normal_initializer(stddev=1))
        self.var = tf.get_variable(shape=[self.M, 1], dtype=tf.float32, name="variance", initializer=tf.ones_initializer())
        self.w =  tf.get_variable(shape=[1, self.M], dtype=tf.float32, name="weight", initializer=tf.random_normal_initializer(stddev=1))
        self.b = tf.Variable(0, dtype=tf.float32, name="bias")

        self.gaussian = tf.exp(tf.negative(tf.divide(tf.square(tf.sub(self.x, self.mu)), self.var)))

        self.yest = tf.add(tf.matmul(self.w, self.gaussian), self.b)

        self.loss = tf.reduce_sum(tf.divide(tf.squared_difference(self.y, self.yest), two))

    def train(self, xs, ys):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.n_epochs):
            for x, y in zip(xs, ys):
                sess.run(optimizer, feed_dict={self.x: x, self.y: y})

    def predict(self, x):
        return sess.run(self.yest, feed_dict={self.x: x})

with tf.Session() as sess:
    model = Model(sess, 20, 0.01, 20)
    model.train(xs, ys)

    fig = plt.figure(figsize=(20,10))
    fig.set_tight_layout({"pad": 1})
    p = plt.subplot(1,2,1)
    xplot = np.linspace(-4, 4, 1000)
    p.plot(xplot, np.sin(xplot), 'b')
    p.plot(xplot, model.predict(xplot)[0], 'r-')
    p.plot(xs, ys, 'go')
    p.set_xlabel('x')
    p.set_ylabel('y', rotation=0)
    p.set_title('Fit')

    p = plt.subplot(1,2,2)
    p.plot(xplot, sess.run(tf.transpose(model.gaussian+model.b), feed_dict={model.x: xplot}))
    p.set_xlabel('x')
    p.set_ylabel('y', rotation=0)
    p.set_title('Bases for fit')

    plt.show()
