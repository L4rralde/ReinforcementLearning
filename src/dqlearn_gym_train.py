import gym

from dqlearn import AtariDeepQLearning


def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    deep_q_agent = AtariDeepQLearning(env)
    deep_q_agent.train()
    deep_q_agent.save('dqln.model')


if __name__ == '__main__':
    main()
