'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

'''
线性回归
一条直线模型产生的点，我戏称这条直线是样本集的母线，一般来说会围绕在母线附近，回归的目的针对当前样本集找出一条最优可能的直线做他们的母线，从概率上将就是使得P(这条直线|这个样本集)的值是最大的
'''

from __future__ import print_function

import tensorflow as tf
import numpy
''' 绘图库，方便的模型、过程和结果可视化 '''
import matplotlib.pyplot as plt

''' numpy提供的随机函数，特别方便，有各种分布类型可选 '''
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
# 写tf.float32最好
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
'''
randn标准正态分布随机发生器，randn() 只产生一个数，randn(1)则产生一个一维数组，只包含一个数, randn(d0, d1, d2, ...)可以产生多维数组
Variable(初始化值)
也可以使用tf.truncated_normal

'''
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
# pred = X * W + b
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
'''
 最小二乘作为成本函数, 找个一个直线的最佳参数组合
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
 Gradient descent
 梯度下降优化器
 也可以尝试AdamOptimizer
 @@Optimizer
  
全部的优化器  
    @@GradientDescentOptimizer
    @@AdadeltaOptimizer
    @@AdagradOptimizer
    @@AdagradDAOptimizer
    @@MomentumOptimizer
    @@AdamOptimizer
    @@FtrlOptimizer
    @@RMSPropOptimizer
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
