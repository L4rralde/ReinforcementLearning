import random

import gym

from qlearn import SarsamaxQLearning as QTable


def main():
	env = gym.make("Taxi-v3", render_mode="human")
	q_table = QTable(env)
	q_table.load_table('FrozenLake-v1_4x4_table.npy')
	mean_reward, std_reward= q_table.evaluate(n_eval_episodes=10, seeds=[random.randint(0,1000) for _ in range(10)])
	print(f"Mean reward: {mean_reward}, standard deviation of reward: {std_reward}")


if __name__ == '__main__':
	main()
