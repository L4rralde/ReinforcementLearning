import copy
from time import sleep

import gym
import numpy as np
import matplotlib.pyplot as plt

import mcts


def policy_player_mcts(mytree, policy_explore: int=100) -> tuple:
	for _ in range(policy_explore):
		mytree.explore()

	next_node = mytree.next()
	next_node.detach_parent()
	return next_node

	
def main() -> None:
	GAME_NAME = "CartPole-v1"
	MAX_ATTEMPTS = 100

	episodes = 10
	rewards = []
	moving_average = []

	for episode in range(episodes):
		print(f"episode #{episode+1}")

		reward_e = 0
		game = gym.make(GAME_NAME, render_mode="rgb_array")
		observation, info = game.reset() #Initial state
		new_game = copy.deepcopy(game)
		mytree = mcts.Node(new_game, False, None, observation, 0) #MCTS

		done = False
		attempt_cnt = 0
		while not done and attempt_cnt<MAX_ATTEMPTS:
			mytree = policy_player_mcts(mytree)
			observation, reward, done, truncated, info = game.step(mytree.action_idx)
			print(f"episode #{episode}, observation={observation}, reward={reward}")
			reward_e += reward
			attempt_cnt += 1
			game.render()
			sleep(1)

		print(f"reward_e: {reward_e}")
		game.close()
		rewards.append(reward_e)
		moving_average.append(np.mean(rewards[-100:]))
	plt.plot(rewards)
	plt.plot(moving_average)
	plt.show()
	print(f"moving_average: {np.mean(rewards[-20:])}")


if __name__ == '__main__':
	main()
