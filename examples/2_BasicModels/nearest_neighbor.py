# - coding: utf-8 -
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

'''
最近邻算法
基本思想：
计算测试对象与训练集中所有对象的距离，找出与他距离最近的那个对象，对应的标签就是我们预测的标签
'''

# __future__是一个编译器兼容性标示，为了后续版本的兼容做出的一个声明，比如这个print_function就是告诉编译器，print方法要以函数的方式对待，不能像现在的版本这样可以不用括号，因为后续的版本已经取消了，不带对号的使用方式
from __future__ import print_function


import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf Graph Input
# None表示不知这个维度有多少 即是一个有784列，但不知道行数的矩阵
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
# L1距离就是 |x - y|，就是两变量相减的绝对值
# 注意也可以写成
# distance = tf.reduce_sum(tf.abs(xtr -xte), reduction_indices = 1)
# 其中xtr和xte的定义你会发现shape不同，怎么回事呢？
# 实验：[ [ 2, 2, 2 ], [ 3, 3, 3] ] - [ 1, 1, 1] = [ [ 1, 1, 1], [ 2, 2, 2 ] ]，明白了吧，就是自动遍历能力，遍历xtr里的每一个第二位元素与xte做差
# reduction_indices 的作用时压缩求和的步骤，比如 redution = 0 就是纵向求和，=1就是横向求和，如果是[ 0, 1 ]就是先纵向求和，再横向求和，对于二维数据而言，得到的就是一个值
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
	# 这里Xte[i,:]直接写成Xte[i]就可以, nn_index返回的就是找到的那个最近对象的索引
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
	# argmax作用就是表示向量转索引 如： [ 0, 0, 1, 0 ] --> 2
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
