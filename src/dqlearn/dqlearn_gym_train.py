import gym
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)
import matplotlib.pyplot as plt

from dqlearn import AtariDeepQLearning


def make_env():
    env = gym.make("Boxing-v4")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

def main():
    env = make_env()
    deep_q_agent = AtariDeepQLearning(env)
    rewards = deep_q_agent.train(total_steps=5e5,
                                learning_start=2e5,
                                buffer_size=int(1e4),
                                exploration_fraction=0.3,
                                gamma=0.95)
    deep_q_agent.save('dqln_boxing.model')
    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    main()
