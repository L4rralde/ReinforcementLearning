import random

import gym
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)

from dqlearn import AtariDeepQLearning



def make_env():
    env = gym.make("Boxing-v4", render_mode="human")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def main():
    env = make_env()
    deep_q_agent = AtariDeepQLearning(env)
    deep_q_agent.load('dqln_boxing.model')
    mean_reward, std_reward= deep_q_agent.evaluate(n_eval_episodes=10, seeds=[random.randint(0,1000) for _ in range(10)])
    print(f"Mean reward: {mean_reward}, standard deviation of reward: {std_reward}")


if __name__ == '__main__':
    main()
