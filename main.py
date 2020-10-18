import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_env
import traci


def run(Env, Epizodes):
    for episode in range(Epizodes):
        Env.reset()
        for step in range(Env.sim_end):
            action = [0, 0]
            new_states, rewards, done, info = Env.step(action)
            # print(new_states)
            if done:
                break

def Multi_Agents_Q_Learning(env, epizodes):
    #Hyperparameters
    alpha = 0.9 #Learning rate
    gamma = 0.99 #Discount rate
    epsilon = 1.0
    max_ep = 1.0
    min_ep = 0.01
    exploration_decay_rate = 0.01

    #For charts
    epzds = []
    cumulative_rewards = []

    # Define Q table list
    Q = []
    for act, obs in zip(env.action_space, env.observation_space):
        action_n = act.n
        num_box = tuple((obs.high + np.ones(obs.shape)).astype(int))
        q_table = np.zeros(num_box + (action_n,))
        Q.append(q_table)

    for episode in range(epizodes):
        states = env.reset()
        total_epizode_rewards = 0
        # for step in range(env.sim_end):
        while traci.simulation.getMinExpectedNumber() > 0:
            action_n = []
            for q_table, state, act in zip(Q, states, env.action_space):
                exp_exp_tradeoff = random.uniform(0, 1)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(q_table[state])
                    action_n.append(action)
                else:
                    action = act.sample()
                    action_n.append(action)

                # if random.uniform(0, 1) < epsilon:
                #     action = act.sample()
                #     action_n.append(action)
                # else:
                #     action = np.argmax(q_table[state])
                #     action_n.append(action)
            new_states, rewards, done, _ = env.step(action_n)
            for q_table, state, action, reward, new_state in zip(Q, states, action_n, rewards, new_states):
                q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])
                # Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            total_epizode_rewards += sum(rewards)
            states = new_states
            # print(step, done)
            if done:
                break
        epsilon = min_ep + (max_ep - min_ep) * np.exp(-exploration_decay_rate * episode)
        cumulative_rewards.append(total_epizode_rewards)
        epzds.append(episode)
    return cumulative_rewards, epzds




if __name__ == "__main__":

    epizodes = []
    rewards_sum = []
    env = gym.make("RL_Traffic-v0")
    n_epizodes = 1


    #------------Modified Algorithm-------------#
    rewards_sum, epizodes = Multi_Agents_Q_Learning(env, n_epizodes)
    plt.plot(epizodes, rewards_sum)
    plt.show()






