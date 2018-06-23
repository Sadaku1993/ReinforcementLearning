#coding:utf-8

"""
Policy-Based Stateから直接最適なActionを予測する

Policy, πθ(s,a)         State sに対してAction aを選択する確率
State Value, Vπθ(s)     State sのValue
Action Value, Qπθ(s,a)  State sでAction aを取った時のValue

目的関数J(θ)はStart Valueと呼ばれ,初期のState s1のValueとなる
J(θ) = Vπθ(s1)

目的関数の勾配
∇θJ(θ)=Eπθ[∇θlogπθ(s,a)Qπθ(s,a)]

"""


import collections
import gym
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical

# to_categorical:one-hotベクトルを作成することができる
# [0, 1, 0, 1, 1, 1] => [[1,0], [0,1], [1,0], [0,1], [0,1], [0,1]]

env = gym.make('CartPole-v1')
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n

LEARNING_RATE = 0.0005


class PolicyEstimator():
    def __init__(self):
        l_input = Input(shape=(NUM_STATES, ))
        l_dence = Dense(20, activation='relu')(l_input)
        action_probs = Dense(NUM_ACTIONS, activation='softmax')(l1_dence)
        model = Model(inputs=[l_input], outputs=[action_probs])
    
        # 状態
        self.state = tf.placeholder(dtype=tf.float32, shape=None)
        # one-hotベクトルで表現したaction
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, NUM_ACTIONS))
        # 割引報酬和
        self.target = tf.placeholder(shape=(None), dtype=tf.float32)
        # モデルから予測した行動選択確率
        self.action_probs = model(self.state)
        # 目的関数J(θ)を最大化することが目的だが、maximizeはないので
        # マイナスをつけてlossの扱いとする
        log_prob = tf.log(tf.reduce_sum(self.action_probs*self.action))
        # 損失関数
        self.loss = -log_prob * self.target

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.minimize = optimizer.minimize(self.loss)   # train_step

    def predict(self, sess, state):
        return sess.run(self.action_probs, feed_dict={self.state: [state]})
    def update(self, sess, state, action, target):
        feed_dict = {self.state: [state], self.target: target, self.action: to_categorical(action, NUM_ACTIONS)}

        sess.run(self.minimize, feed_dict = feed_dict)
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

policy_estimator = PolicyEstimator()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

gamma = 1.0
num_episodes = 10000

Step = collections.namedtuple("Step", ["state", "action", "reward"])
last_100 = np.zeros(100)


for i_episode in range(1, num_episodes+1):
    state = env.reset()

    episode = []
    temp_loss = []
    sum_reward = 0
    while True:
        action_probs = policy_estimator.predict(sess, state)[0]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)

        episode.append(Step(state=state, action=action, reward=reward))

        if done:
            break

        state = next_state
    
    loss_list = []

    for t, step in enumerate(episode):
        # t stepから行動終了までの割引報酬和を算出
        target = sum(gamma**i * t2.reward for i, t2 in enumerate(episode[t:]))
        # 損失を計算
        loss = policy_estimator.update(sess, step.state, step.action, target)
        loss_list.append(loss)

    # log
    total_reward = sum(e.reward for e in episode)
    last_100[i_episode % 100] = total_reward
    last_100_avg = sum(last_100) / (i_episode if i_episode < 100 else 100)
    avg_loss = sum(loss_list) / len(loss_list)
    print('episode %s avg_loss %s reward: %d last 100: %f' % (i_episode, avg_loss, total_reward, last_100_avg))
    
    if last_100_avg >= env.spec.reward_threshold:
        break

for i_episode in range(10):
    state = env.reset()
    r_sum = 0
    while True:
        action_probs = policy_estimator.predict(sess, state)[0]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)
    
        if done:
            break
    
        state = next_state
        r_sum += reward

    print "Test:%d Reward:%d" %(i_episode, r_sum)
