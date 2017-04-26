import os
import tensorflow as tf

#v1 = tf.Variable(tf.ones(100))
#v2 = tf.Variable(tf.zeros(100))
#v1 = tf.Variable(tf.zeros(100))
#v2 = tf.Variable(tf.ones(100))
#with tf.variable_scope("test"):
with tf.variable_scope("test_3"):
    #v1 = tf.Variable(tf.zeros(100))
    #v2 = tf.Variable(tf.ones(100))

    v1 = tf.Variable(tf.ones(100))
    v2 = tf.Variable(tf.zeros(100))

#new_name1 = v1.op.name
#new_name2 = v2.op.name
#new_name1 = 'test_2' + new_name1[6:]
#new_name2 = 'test_2' + new_name2[6:]
#print(new_name1)
#print(new_name2)

v3 = tf.Variable(tf.ones(200))
v4 = tf.Variable(tf.zeros(200))

save_prefix = '/mnt/fs0/chengxuz/test'

#saver = tf.train.Saver({'v1': v1, 'v2': v2})
saver = tf.train.Saver()
#saver = tf.train.Saver([v1, v2])
#saver = tf.train.Saver({new_name1: v1, new_name2: v2})
#with tf.variable_scope("test_3"):
#    saver = tf.train.Saver()

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

v1_before = sess.run(v1)
print(v1_before)

#saver.save(sess, os.path.join(save_prefix, 'test_save'), global_step=300)
saver.restore(sess, tf.train.latest_checkpoint(save_prefix))

v1_after = sess.run(v1)
print(v1_after)

init_op = tf.global_variables_initializer()
sess.run(init_op)

v1_init = sess.run(v1)
print(v1_init)

sess.close()
