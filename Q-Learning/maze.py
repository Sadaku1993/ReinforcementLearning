"""
Reinforcement learning maze

Red rectangle:          agent
Black rectangle:        hells
Yellow bin circle:      goal
All other states:       ground

"""

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import math

class Maze(object):
    def __init__(self, size):
        # start position
        self.start = np.array([5, 5])
        # goal position
        self.goal = np.array([15,15])
        # sub goal
        self.sub_goal_x = []
        self.sub_goal_y = []
        self.sub_goal_reward = []
        # obstacle position
        self.ox = []
        self.oy = []
        # Grid World Size 
        self.size = size
        # State
        self.state = np.array([5, 5])
        self.path_x = [5]
        self.path_y = [5]
    def obstacle(self):
        for i in range(self.size):
            self.ox.append(i)
            self.oy.append(0)
        for i in range(self.size):
            self.ox.append(self.size)
            self.oy.append(i)
        for i in range(self.size+1):
            self.ox.append(i)
            self.oy.append(self.size)
        for i in range(self.size+1):
            self.ox.append(0)
            self.oy.append(i)
        for i in range(5):
            self.ox.append(10)
            self.oy.append(i)
        for i in range(5):
            self.ox.append(10)
            self.oy.append(self.size - i)

    def sub_goal(self, x, y, reward):
        self.sub_goal_x.append(x)
        self.sub_goal_y.append(y)
        self.sub_goal_reward.append(reward)

    def show(self, time):
        fig = plt.figure()
        plt.plot(self.ox, self.oy, ".k")
        plt.plot(self.start[0], self.start[1], "xr")
        plt.plot(self.goal[0],  self.goal[1],  "xb")
        plt.grid(True)
        plt.axis("equal")
        plt.plot(self.state[0], self.state[1], "xg")
        plt.plot(self.path_x, self.path_y, "-r")
        plt.pause(time)
        plt.close(fig)


    def calc_collision(self):
        collision = False
        for iox, ioy in zip(self.ox, self.oy):
            if self.state[0] == iox-1 and self.state[1] == ioy-1:
                collision = True
                break
        return collision

    def step(self, action):
        if action == 0:
            self.state[1] += 1
        if action == 1:
            self.state[0] += 1
        if action == 2:
            self.state[1] -= 1
        if action == 3:
            self.state[0] -= 1
        if action == 4:
            self.state[0] += 1
            self.state[1] += 1
        if action == 5:
            self.state[0] += 1
            self.state[1] -= 1
        if action == 6:
            self.state[0] -= 1
            self.state[1] -= 1
        if action == 7:
            self.state[0] -= 1
            self.state[1] += 1
    
        self.path_x.append(self.state[0])
        self.path_y.append(self.state[1])
        index = self.state[0]+self.state[1]*self.size

        if int(np.sum(self.goal-self.state)) == 0:
            print("Goal")
            reward = 10
            done = True

        elif self.calc_collision():
            print("Collosion")
            reward = -10
            done = True

        else:
            distance = np.sqrt((self.state[0]-self.goal[0])**2+(self.state[1]-self.goal[1])**2)
            maxD = np.sqrt(self.goal[0]**2+self.goal[1]**2)
            distance /= maxD
            distance *= -1
            distance += 1
            reward = distance*0.1
            # reward = -0.01
            done = False
        
        # print("--- State:%d %d Index:%d" % (self.state[0], self.state[1], index))
        return index, reward, done

    def reset(self):
        self.path_x = [5]
        self.path_y = [5]
        self.state = np.array([5, 5])
        return self.state[0]+self.state[1]*self.size
        

if __name__ == '__main__':
    env = Maze(20)
    env.obstacle()
    for i in range(50):
        action = np.random.randint(0,8)
        index, reward, done = env.step(action)
        if done:
            break
    env.show(1)
