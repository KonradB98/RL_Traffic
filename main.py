import random
import numpy as np
from Enviroment import TrafficEnv

def QLearning(Env, Epizodes):
    actions = len(Env.getLightActions()) #Number of available actions (4 for Two way intersection)
    states = Env.sim_end #Number of simulation states
    Q = np.zeros((states, actions)) #Initialize Q table with zeros

    alpha = 0.85 #Learning rate
    gamma = 0.8 #Discount rate
    epsilon = 1.0
    max_ep = 1.0
    min_ep = 0.1
    exploration_decay_rate = 0.05

    cumulative_rewards = []
    for episode in range(Epizodes):
        Env.startSumo()
        state = 0
        total_epizode_rewards = 0
        for step in range(Env.sim_end):
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])
            else:
                action = random.randint(0, 3)
            new_state, reward = Env.step(action)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            # Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]))
            total_epizode_rewards += reward
            state = new_state
        Env.stopSumo()
        epsilon = min_ep + (max_ep - min_ep) * np.exp(-exploration_decay_rate * episode)
        cumulative_rewards.append(total_epizode_rewards)

    print(cumulative_rewards)


if __name__ == "__main__":
    env = TrafficEnv()
    n_epizodes = 100
    QLearning(env, n_epizodes)






