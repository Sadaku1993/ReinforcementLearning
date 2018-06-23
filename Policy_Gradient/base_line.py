#coding:utf-8

"""
PolicyBased:分散が高いのが弱点
            |
            |
            v
BaseLineを用いて分散を抑える
BaseLineにはStateValueを導入するのがいい
Advantage A = Action Value V - State Value V
を利用するのがいい
"""


import collections
import gym
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical

env = gym.make('CartPole-v1')

NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n

LEARNING_RATE_POLICY = 0.0005
LEARNING_RATE_VALUE = 0.0005

class PolicyEstimator():
    def __init__(self):
        with tf.variable_scope('policy_estimator'):
            l_input = Input(shape=(NUM_STATES, ))
            l_dense = Dense(16, activation='relu')(l_input)
            action_probs = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
            policy_network = Model(input=l_input, output=action_probs)
            
            # 状態 
            self.state = tf.placeholder(tf.float32)
            # one-hotベクトルで表現したaction
            self.action = tf.placeholder(shape=(None, NUM_ACTIONS), dtype=tf.float32)
            # 割引報酬和
            self.target = tf.placeholder(shape=(None), dtype=tf.float32)
            # モデルから予測した行動選択確率
            self.action_probs = policy_network(self.state)

            # 目的関数J(θ)を最大化することが目的だが、maximizeはないので
            # マイナスをつけてlossの扱いとする
            log_prob = tf.log(tf.reduce_sum(self.action_probs * self.action))
            self.loss = -log_prob * self.target

            optimizer = tf.train.AdamOptimizer(LEARNING_RATE_POLICY)
            self.minimize  = optimizer.minimize(self.loss)
    
    def predict(self, sess, state):
        return sess.run(self.action_probs, feed_dict={self.state: [state]})

    def update(self, sess, state, action, target):
        feed_dict = {self.state: [state], self.target: target, self.action: to_categorical(action, NUM_ACTIONS)}
        sess.run(self.minimize, feed_dict=feed_dict)
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

class ValueEstimator():
    def __init__(self):
        with tf.variable_scope('value_estimator'):
            l_input = Input(shape=(NUM_STATES, ))
            l_dense = Dense(16, activation='relu')(l_input)
            state_value = Dense(1, activation='linear')(l_dense)
            value_network = Model(input=l_input, output=state_value)

            self.state = tf.placeholder(tf.float32)
            self.action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
            self.target = tf.placeholder(tf.float32, shape=(None))
            self.state_value = value_network(self.state)[0][0]
            self.loss = tf.squared_difference(self.state_value, self.target)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE_VALUE)
            self.minimize = self.optimizer.minimize(self.loss)

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32)
        action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        target = tf.placeholder(tf.float32, shape=(None))

        state_value = model(state)[0][0]
        loss = tf.squared_difference(state_value, target)

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE_VALUE)
        minimize = optimizer.minimize(loss)

        return state, action, target, state_value, minimize, loss

    def predict(self, sess, state):
        return sess.run(self.state_value, { self.state: [state] })

    def update(self, sess, state, target):
        feed_dict = {self.state:[state], self.target:target}
        _, loss = sess.run([self.minimize, self.loss], feed_dict)
        return loss

policy_estimator = PolicyEstimator()
value_estimator  = ValueEstimator()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

gamma = 1.0
num_episodes = 1000

Step = collections.namedtuple("Step", ["state", "action", "reward"])
last_100 = np.zeros(100)

for i_episode in range(1, num_episodes+1):
    state = env.reset()

    episode = []
    
    while True:
        action_probs = policy_estimator.predict(sess, state)[0]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)
        episode.append(Step(state=state, action=action, reward=reward))

        if done:
            break

        state = next_state

    loss_p_list = []
    loss_v_list = []

    # # step数が500以下は失敗
    failed = len(episode) < 500

    for t, step in enumerate(episode):
        total_return = sum(gamma**i * t2.reward for i, t2 in enumerate(episode[t:]))
        baseline_value = value_estimator.predict(sess, step.state)
        advantage = total_return - baseline_value
            
        # failの時のみ学習を行う
        if failed:
            loss_v = value_estimator.update(sess, step.state, total_return)
        loss_p = policy_estimator.update(sess, step.state, step.action, advantage)

        loss_p_list.append(loss_p)
        loss_v_list.append(loss_v)

    total_reward = sum(e.reward for e in episode)
    last_100[i_episode % 100] = total_reward
    last_100_avg = sum(last_100) / (i_episode if i_episode < 100 else 100)
    avg_loss_p = sum(loss_p_list) / len(loss_p_list)
    avg_loss_v = sum(loss_v_list) / len(loss_v_list)
    print('episode %s p: %s v: %s reward: %d last 100: %f' % (i_episode, avg_loss_p, avg_loss_v, total_reward, last_100_avg))
    
    if last_100_avg >= env.spec.reward_threshold:
        break
