from maze import *

def get_action(q_table, next_state, epsilon):
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.randint(0, 8)
    return next_action

def update_Qtable(q_table,state,action,reward,next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q = np.amax(q_table[next_state])
    q_table[state, action] = (1-alpha) * q_table[state,action]+ alpha * (reward + gamma * next_Max_Q)
    return q_table

def calc_epsilon(episode):
    epsilon = 1.0  - 0.001*episode
    if epsilon < 0.1:
        epsilon = 0.1
    return epsilon

def main():
    size = 20
    env = Maze(size)
    env.obstacle()
    num_episodes = 10000
    max_number_of_steps = 300
    num_action = 8
    q_table = np.random.uniform(low = -1, high=1, size=(size**2, num_action))
    print(q_table.shape)
    is_learned = 0

    for episode in range(num_episodes):
        state = env.reset()
        action = np.argmax(q_table[state]) 
        episode_reward = 0
        epsilon = calc_epsilon(episode)
        print("Episode:%d Epsilon:%f" % (episode, epsilon))
        for t in range(max_number_of_steps):
            state_, reward, done = env.step(action)
            episode_reward += reward
            q_table = update_Qtable(q_table, state, action, reward, state_)
            action = get_action(q_table, state_, epsilon)
            state = state_
            
            if done:
                print("Result --> Step:%d Reward:%.2f" % (t+1, episode_reward))
                break

            if t == max_number_of_steps - 1:
                print("Result --> Fail to arrive Goal")
                
        if (episode % 100 == 0):
            env.show(.5)
        print("")

if __name__ == "__main__":
    main()
