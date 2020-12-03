import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_env
import traci
from numpy import save
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter

import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


##---------------Stable Baselines Callback---------------##
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.model_path = os.path.join(log_dir, 'model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 1000 episodes
              mean_reward = np.mean(y[-1:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                self.model.save(self.model_path)

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True



def run(Env, Epizodes):
    for episode in range(Epizodes):
        o = Env.reset()
        for step in range(Env.sim_end):
            action = 0
            new_state, reward, done, _ = Env.step(action)
            # print(new_states)
            if done:
                break


def Q_learning(env, epizodes):
    #Hyperparameters
    alpha = 0.69 #Learning rate
    gamma = 0.97 #Discount rate
    epsilon = 1.0
    max_ep = 1.0
    min_ep = 0.1
    exploration_decay_rate = 0.00075

    #For charts
    epzds = []
    cumulative_rewards = []

    nb = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(nb + (env.action_space.n,))

    for episode in range(epizodes):
        state = env.reset()
        total_epzd_reward = 0
        for step in range(env.sim_end):

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            new_state, reward, done, _ = env.step(action)

            total_epzd_reward += reward

            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])

            # total_epizode_rewards += sum(rewards)
            state = new_state
            # print(step, done)
            if done:
                break
        print("Episode - " + repr(episode) + " - reward:")
        # total_epzd_reward /= env.sim_end
        print(total_epzd_reward)
        # if epsilon >= min_ep:
        #     epsilon -= exploration_decay_rate
        # if not epsilon <= 0.05:
        #     epsilon *= exploration_decay_rate
        epsilon = min_ep + (max_ep - min_ep) * np.exp(-exploration_decay_rate * episode)
        cumulative_rewards.append(total_epzd_reward)
        epzds.append(episode)
    return cumulative_rewards, epzds, q_table


def test_qlearning(env, episodes, q_table):
    for i in range(episodes):
        state = env.reset()
        for step in range(env.sim_end):
            action = np.argmax(q_table[state])
            new_state, reward, done, _ = env.step(action)
            state = new_state
            if done:
                break


def static_agent(env, episodes):
    for i in range(episodes):
        env.reset()
        action = 0
        time = 0
        for step in range(env.sim_end):
            time += 1
            if time == 30 and action == 0:
                time = 0
                action = 1
            elif time == 5 and action == 1:
                time = 0
                action = 2
            elif time == 30 and action == 2:
                time = 0
                action = 3
            elif time == 5 and action == 3:
                time = 0
                action = 0
            new_state, reward, done, _ = env.step(action)
            if done:
                break



if __name__ == "__main__":

    env = gym.make("RL_Traffic-v3")
    # check_env(env)
    ##----------------------------------LEARNING-------------------------------------##

    # log_dir = "DQN_HV2/"
    # os.makedirs(log_dir, exist_ok=True)
    # env = Monitor(env, log_dir)
    # model = DQN('MlpPolicy', env, learning_rate=0.0001, prioritized_replay=False, verbose=1)
    # time_steps = 1550000
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # model.learn(total_timesteps=int(time_steps), callback=callback)
    # model.save("dqn_hv2")

    ##--------------------------------CONTINUE LEARNING----------------------------------##

    # log_dir = "DQN_TE2/"
    # os.makedirs(log_dir, exist_ok=True)
    # env = Monitor(env, log_dir)
    # loaded_model = DQN.load("dqn_old_wt.zip", env=env)
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # loaded_model.learn(total_timesteps=int(time_steps), callback=callback)
    # loaded_model.save("dqn_4speed_v2_2")

    ##------------------------------------TEST---------------------------------------##

    epizodes = 1
    loaded_model = DQN.load("agent_dqn_hv.zip", env=env)
    for j in range(epizodes):
        obs = env.reset()
        for i in range(env.sim_end):
            action, _states = loaded_model.predict(obs)
            obs, rewards, dones, info = env.step(action)


    ##------------------------------------Plotting---------------------------------------##

    # log_dir = "DQN_HV2/"
    # time_steps = 1550000
    # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
    # plt.show()

    # #-------------------------------CLASSIC QLEARNING--------------------------------------#

    # n_epizodes = 1550
    # rewards_sum, epizodes, Q = Q_learning(env, n_epizodes)
    # # test_qlearning(env, n_epizodes, q_table)
    # save('ql_wt.npy', Q)
    # f1 = open('rewards.txt', 'w')
    # for i in rewards_sum:
    #     f1.write(str(i) + '\n')
    # f1.close()
    # f2 = open('epizodes.txt', 'w')
    # for i in epizodes:
    #     f2.write(str(i) + '\n')
    # f2.close()
    #
    # plt.plot(epizodes, rewards_sum)
    # plt.show()

    ##-----------------------TEST Q-learning-------------------##

    # n = 100
    # Q = np.load("ql_wt.npy")
    #
    # test_qlearning(env, n, Q)

    # plt.plot(epizodes, rewards_sum)
    # plt.show()

    #---------------Static Light--------------#

    # epzd = 100
    #
    # static_agent(env, epzd)







