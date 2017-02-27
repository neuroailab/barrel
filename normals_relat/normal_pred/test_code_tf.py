import tensorflow as tf
import numpy as np

test_a = np.random.rand(10, 3)
test_b = np.random.rand(10, 3)
print(test_a)
print(test_b)

a = tf.constant(test_a, shape=[10, 3], name='a')
b = tf.constant(test_b, shape=[10, 3], name='b')
c = tf.nn.l2_normalize(a, 1)
d = tf.reduce_sum(tf.multiply(a,b))

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print sess.run(c)
print sess.run(d)
#print np.dot(test_a, test_b)
print np.dot(test_a.reshape(test_a.size), test_b.reshape(test_b.size))
