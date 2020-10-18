from gym.envs.registration import register

register(
    id='RL_Traffic-v0',
    entry_point='gym_env.envs:TrafficEnv',
    max_episode_steps=10000,
)