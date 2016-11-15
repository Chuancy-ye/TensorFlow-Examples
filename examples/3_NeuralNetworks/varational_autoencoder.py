# -*- coding: utf-8 -*-

""" Varational Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

变分自动编码器
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 样本集X
n_input = 784 # 28 * 28
X = tf.placeholder(tf.float32, [ None, n_input ])

# Encoder

## \mu(X)
W_mu =
b_mu = 
mu = tf.matmul(X, W_mu) + b_mu

## \Sigma(X)
W_sigma = 
b_sigma =
sigma = tf.matmul(X, W_sigma) + b_sigma

## KLD = D[N(mu(X), sigma(X))||N(0, I)] = 1/2 * sum(sigma_i + mu_i^2 - log(sigma_i) - 1)
KLD = 0.5 * tf.reduce_sum(sigma + tf.pow(mu, 2) - tf.log(sigma) - 1, reduction_indices = 1) # reduction_indices = 1代表按照每个样本计算一条KLD


# epsilon = N(0, I) 采样模块
epsilon = tf.random_normal(tf.shape(sigma), name = 'epsilon')

# z = mu + sigma^ 0.5 * epsilon
z = mu + tf.pow(sigma, 0.5) * epsilon

# Decoder ||f(z) - X|| ^ 2 重建的X与X的欧式距离，更加成熟的做法是使用crossentropy
W_decoder
b_decoder
reconstructed_X = tf.matmul(z, W_decoder) + b_decoder


reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(reconstructed_X, X), reduction_indices = 1)

loss = tf.reduce_mean(reconstruction_loss + KLD)


# minimize(loss)
