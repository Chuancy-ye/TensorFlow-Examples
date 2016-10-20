import tensorflow as tf

a = tf.placeholder(tf.int32, [None, 4])
b = tf.placeholder(tf.int32, [4])

c = a - b

feed = {
	a: [ [ 2, 2, 2, 2 ], [ 3, 3, 3, 3 ], [ 4, 4, 4, 4 ] ],
	b: [ 1, 1, 1, 1 ]
}
sess = tf.Session()
print(sess.run(c, feed_dict = feed))

print 'reduce to 1'
e = tf.reduce_sum(c, reduction_indices = 1)
print(sess.run(e, feed_dict = feed))

print 'reduce to 0'
d = tf.reduce_sum(c, reduction_indices = 0)
print(sess.run(d, feed_dict = feed))


print 'reduce to [0, 1]'
d = tf.reduce_sum(c, reduction_indices = [0, 1])
print(sess.run(d, feed_dict = feed))

print 'reduce to [1, 0]'
d = tf.reduce_sum(c, reduction_indices = [1, 0])
print(sess.run(d, feed_dict = feed))

print 'reduce to [0, 0]'
d = tf.reduce_sum(c, reduction_indices = [0, 0])
print(sess.run(d, feed_dict = feed))

print 'reduce to [1, 1]'
d = tf.reduce_sum(c, reduction_indices = [1, 1])
print(sess.run(d, feed_dict = feed))
