import numpy as np
rng = np.random
print rng.randn()
print rng.randn(1)
print rng.randn(5)
print rng.randn(1, 1)
print rng.randn(2, 3)

import tensorflow as tf
# illegal print tf.truncated_normal()
# illegal print tf.truncated_normal(1)
print tf.truncated_normal([ 1 ])
print tf.truncated_normal([ 2, 2 ])

