#coding:utf-8

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return weight

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return bias

def fully_connected_relu(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.relu(layer)

def fully_connected_softmax(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.softmax(layer)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            weight_1 = init_weight(shape=[n_features, 20], st_dev=1.0)
            bias_1 = init_bias(shape=[20], st_dev=0.1)
            layer_1 = fully_connected_relu(self.s, weight_1, bias_1)

            weight_2 = init_weight(shape=[20, n_actions], st_dev=1.0)
            bias_2   = init_bias(shape=[n_actions], st_dev=0.1)
            self.acts_prob  = fully_connected_softmax(layer_1, weight_2, bias_2)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            weight_1 = init_weight(shape=[n_features, 20], st_dev=1.0)
            bias_1   = init_bias(shape=[20], st_dev=0.1)
            layer_1  = fully_connected_relu(self.s, weight_1, bias_1)

            weight_2 = init_weight(shape=[20, 1], st_dev=1.0)
            bias_2   = init_weight(shape=[1], st_dev=0.1)
            self.v   = tf.add(tf.matmul(layer_1, weight_2), bias_2)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

sess = tf.Session()

actor_network = Actor(sess=sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic_network = Critic(sess=sess, n_features=N_F, lr=LR_C)

init = tf.global_variables_initializer()
sess.run(init)

last_100 = np.zeros(100)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor_network.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic_network.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor_network.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            if i_episode == 0:
                last_100[i_episode%100] = 0

            else:
                last_100[i_episode%100] = sum(track_r)
            last_100_avg = sum(last_100) / (i_episode if i_episode < 100 else 100)
            if last_100_avg > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", last_100_avg)
            break
