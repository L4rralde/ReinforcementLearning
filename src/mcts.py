import random
import copy
import math

class Node:
	def __init__(self, game, done, parent, observation, action_idx) -> None:
		self.child = None #Child nodes
		self.T = 0 #Represents the sum of the value of the rollouts that have been started from this node
		self.N = 0 #Visit count
		self.game = game #The environment
		self.observation = observation #Current state of the game in the node
		self.done = done #Game is done
		self.parent = parent #Parent node. For backpropagation
		self.action_idx = action_idx #Index of the action that lead to this node

		self.c = 1.0 #FOR UCB score???

	def getUCBscore(self) -> float:
		"""
		Upper Confidence Bound
		"""
		if self.N == 0:
			return float('inf')

		if self.parent:
			top_node = self.parent
		else:
			top_node = self

		return (self.T/self.N)+self.c*math.sqrt((math.log(top_node.N)/self.N))

	def create_child(self) -> None:
		if self.done:
			return

		child = {}
		for action in range(self.game.action_space.n):
			game = copy.deepcopy(self.game)
			observation, reward, done, truncated, info = game.step(action)
			child[action] = Node(game, done, self, observation, action)

		self.child = child

	def explore(self) -> None:
		"""
		The search algorithm is as follows:
		- From the current node, recursively pick the children which maximizes the value
		- When a leaf is reached:
			- If it has never been explored before, do a rollout and update its current value
			- Otherwise, expand the node creating its children, randomly pick one, do a rollout and update its value
		- Backpropagate
		"""
		current = self
		while current.child:
			child = current.child
			max_U = max(children.getUCBscore() for children in child.values())
			actions = [a for a,c in child.items() if c.getUCBscore()==max_U]
			if len(actions) == 0:
				raise ValueError(f"Error zero length: {max_U}")
			action = random.choice(actions)
			current = child[action]

		#Play a random game, or expand if needed
		if current.N < 1:
			current.T += current.rollout()
		else:
			current.create_child()
			if current.child:
				current = random.choice(current.child)
			current.T += current.rollout()

		#Update statistics and backpropagate
		current.N += 1
		parent = current
		while parent.parent:
			parent = parent.parent
			parent.N += 1
			parent.T += current.T

	def rollout(self) -> int:
		if self.done:
			return 0

		total_reward = 0
		new_game = copy.deepcopy(self.game)		
		done = False
		while not done:
			action = new_game.action_space.sample()
			observation, reward, done, truncated, info = new_game.step(action)
			total_reward += reward
			if done:
				new_game.reset()
				new_game.close()
				break

		return total_reward

	def next(self):
		"""
		Ask for the next action to play from the current node
		"""
		if self.done:
			raise ValueError("Game over")

		if not self.child:
			raise ValueError("No children found and game is not over")

		child = self.child
		max_N = max(node.N for node in child.values())
		max_children = [c for a,c in child.items() if c.N == max_N]

		if len(max_children) == 0:
			raise ValueError(f"Error zero length: {max_children}")

		max_child = random.choice(max_children)

		return max_child

	def detach_parent(self) -> None:
		"""free memory detaching nodes"""
		del self.parent
		self.parent = None
