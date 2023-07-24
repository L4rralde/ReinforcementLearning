import random

import gym
from dqlearn import AtariDeepQLearning


def main():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    deep_q_agent = AtariDeepQLearning(env)
    deep_q_agent.load('dqln.model')
    mean_reward, std_reward= deep_q_agent.evaluate(n_eval_episodes=100, seeds=[random.randint(0,1000) for _ in range(10)])
    print(f"Mean reward: {mean_reward}, standard deviation of reward: {std_reward}")


if __name__ == '__main__':
    main()
