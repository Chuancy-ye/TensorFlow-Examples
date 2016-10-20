#coding:utf-8
'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
多层感知器模型应该是这几个文件中的第一个应该学习的文件，是最早提出来的深度神经网络
简称MLP
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    ''' x * weights_h1 + b1 
        relu做激活函数, 计算量小，效果也很好
    '''
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    ''' layer_1 * weights_h2 + b2 '''
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    ''' layer_2 * weight_out + out '''
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
''' 随机初始化权重 '''
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
'''
softmax_cross_entropy_with_logits详解

softmax的效果就是将输入的数据压扁以使得sum(outputs) = 1
如:
a = tf.constant(np.array([[.1, .3, .5, .9]]))
print s.run(tf.nn.softmax(a))
[[ 0.16838508  0.205666    0.25120102  0.37474789]]
相反：softmax_cross_entropy_with_logits计算softmax处理后的值的交叉熵
mid = softmax(inputs)
outputs = cross_entropy(mid)

The cross entropy is a summary metric - it sums across the elements. The output of tf.nn.softmax_cross_entropy_with_logits on a shape [2,5] tensor is of shape [2,1] (the first dimension is treated as the batch).
交叉熵是一种系统的度量标准，总结整个系统内元素之间的某种关系。
shape是[2, 5]的向量通过softmax_cross_entropy_with_logits计算后得到的是一个[2, 1]的向量，第一个维度默认当做batch对待

logits代表没有正则化的数据，输出之和一般也不等于1，不能解释为概率，所以定义softmax(logits)的输入就是logits，logits通过softmax处理后就变成了一种概率表示，加和为1

softmax是一个计算函数，softmax_cross_entropy_with_logits是一个训练中才使用的softmax的成本计算函数
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

'''
Adam优化器比梯度下降优化器要快
'''
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    # 做15次迭代训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
