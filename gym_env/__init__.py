from gym.envs.registration import register

register(
    id='RL_Traffic-v3',
    entry_point='gym_env.envs:TrafficEnv',
    max_episode_steps=5000,
)