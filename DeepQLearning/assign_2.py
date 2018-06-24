#coding:utf-8

"""
tf.assignの使い方の確認
Pattern1 Pattern2どちらでもOK
"""

import tensorflow as tf
import numpy as np

# Pattern1
# with tf.variable_scope("target_vars"):
#     x_ = tf.get_variable("x_", [5], initializer=tf.constant_initializer(value=0), collections=['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
# with tf.variable_scope("predict_vars"):
#     y_ = tf.get_variable("y_", [5], initializer=tf.constant_initializer(value=1), collections=['predict_net_params', tf.GraphKeys.GLOBAL_VARIABLES])

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# x_params = tf.get_collection("target_net_params")
# y_params = tf.get_collection("predict_net_params")

#Pattern2
with tf.variable_scope("target_vars"):
    x_ = tf.get_variable("x_", [5], initializer=tf.constant_initializer(value=0))
with tf.variable_scope("predict_vars"):
    y_ = tf.get_variable("y_", [5], initializer=tf.constant_initializer(value=1))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_params = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="target_vars")
y_params = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="predict_vars")

print(x_)
print(y_)
print("x_params", x_params)
print("y_params", y_params)
print(sess.run(x_))
print(sess.run(y_))

replace_target_op = [tf.assign(x, y) for x, y in zip(x_params, y_params)]
sess.run(replace_target_op)

# for x, y in zip(x_params, y_params):
#     assign_op = x.assign(y)
#     sess.run(assign_op)

print(sess.run(x_))
print(sess.run(y_))
