import random
from abc import ABC
from tqdm import tqdm

import numpy as np


class QLearning(ABC):
	def __init__(self, env) -> None:
		self.env = env
		self.table = self.init_q_table()

	def init_q_table(self):
		pass

	def acting_policy(self):
		pass

	def updating_policy(self):
		pass

	def train(self):
		pass


class SarsamaxQLearning(QLearning):
	def init_q_table(self) -> np.ndarray:
		return np.zeros((self.env.observation_space.n, self.env.action_space.n))

	def acting_policy(self, state: int) -> int:
		action = np.argmax(self.table[state])
		return action

	def updating_policy(self, state: int, epsilon: float) -> int:
		if epsilon > random.random():
			action = random.randint(0, self.env.action_space.n-1)
		else:
			action = self.acting_policy(state)
		return action

	def train(	self,
				n_training_episodes: int = 10000,
				alpha: float = 0.7, #Learning rate
				max_steps: int = 99, #Max steps per episode
				gamma: float = 0.95, #Discounting rate
				min_epsilon: float = 0.05, #Min exploration probability
				decay_rate: float = 0.0005) -> None:	
		print("Training")
		for episode in tqdm(range(n_training_episodes)):
			state, info = self.env.reset()
			epsilon = min_epsilon + (1.0-min_epsilon)*np.exp(-decay_rate*episode)
			for step in range(max_steps):
				action = self.updating_policy(state, epsilon)
				new_state, reward, done, truncated, info = self.env.step(action)
				self.table[state][action] = self.table[state][action]\
											+ alpha*(reward+gamma*np.max(self.table[new_state])\
														- self.table[state][action])
				state = new_state
				if done:
					break

	def evaluate(	self,
					max_steps: int = 99,
					n_eval_episodes: int = 100,
					seeds: list = []) -> tuple:

		episode_rewards = []
		for episode in tqdm(range(n_eval_episodes)):
			if seeds:
				state, info = self.env.reset(seed=seeds[episode])
			else:
				state, info = self.env.reset()

			total_rewards_e = 0
			for step in range(max_steps):
				action = self.acting_policy(state)
				new_state, reward, done, truncated, info = self.env.step(action)
				total_rewards_e += reward

				if done or truncated:
					break
				state = new_state
			episode_rewards.append(total_rewards_e)
		mean_reward = np.mean(episode_rewards)
		std_reward = np.std(episode_rewards)

		return mean_reward, std_reward

	def save_table(self, fname: str) -> None:
		with open(fname, 'wb') as f:
			np.save(f, self.table)

	def load_table(self, fname: str) -> None:
		with open(fname, 'rb') as f:
			self.table = np.load(f)
