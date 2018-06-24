#coding:utf-8
import tensorflow as tf
import numpy as np

x = tf.Variable(tf.zeros([5]))
y = tf.Variable(tf.ones([5]))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# x, yの値を表示
print(x.name, sess.run(x))
print(y.name, sess.run(y))

input_placeholder = tf.placeholder(tf.float32, shape=[5])
assign_op = x.assign(input_placeholder)
print(sess.run(assign_op, feed_dict={input_placeholder: np.ones(5).astype(np.float32)}))
