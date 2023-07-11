import gym
from qlearn import SarsamaxQLearning as QTable


def main():
	env = gym.make("Taxi-v3", render_mode="rgb_array")
	q_table = QTable(env)
	q_table.train()
	q_table.save_table('FrozenLake-v1_4x4_table.npy')
	mean_reward, std_reward= q_table.evaluate(n_eval_episodes=1)
	print(f"Mean reward: {mean_reward}, standard deviation of reward: {std_reward}")


if __name__ == '__main__':
	main()
