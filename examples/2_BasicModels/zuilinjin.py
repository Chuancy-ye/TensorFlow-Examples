from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

train_image_set, train_label_set = mnist.train.next_batch(500)
test_image_set, test_label_set = mnist.test.next_batch(10)


# placeholder, Graph Input
train_images = tf.placeholder(tf.float32, [ None, 784 ])
test_image = tf.placeholder(tf.float32, [ 784 ])


# L1 distance
# the - will adapts different dimensions and than will iterates the train_images to minus test_image one by one
distance = tf.reduce_sum(tf.abs(train_images - test_image), reduction_indices = 1)

# find the nearest training example, Graph, not np.argmin
predict_index = tf.arg_min(distance, 0)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

accuracy = 0
for i in range(len(test_image_set)):
	best_neighbour_index = sess.run(predict_index, feed_dict={ train_images: train_image_set, test_image: test_image_set[i] })

	#print("test", i, "Prediction:", train_label_set[best_neighbour_index], "True Class:", test_label_set[i])
	# vector representation --> index e.g. [ 0, 1, 0, 0] --> 1, [ 0, 0, 0, 1 ] --> 3	
	print("test", i, "Prediction:", np.argmax(train_label_set[best_neighbour_index]), "True Class:", np.argmax(test_label_set[i]))
	if np.argmax(train_label_set[best_neighbour_index]) == np.argmax(test_label_set[i]):
		accuracy += 1. / len(test_image_set)

print("Done!", "Accuracy:", accuracy)
