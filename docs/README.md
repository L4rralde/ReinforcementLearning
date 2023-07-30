# My reinforcement learning journey

This is my first project while being a full time engineer. So far I have implemented Q-Learning, Deep Q-Learning, and Monte Carlo Tree Search or MCTS algorithms. I tested them with some games from openAI's gym module.

Thanks! I wanted you to know that I really appreciate that you explore this repo.

## Q-Learning algorithm
Where Q stands for Quality. My implementation is based on the Sarsamax (Q-Learning) algorithm depicted in the following figure.

<div align="center">
<img src="https://github.com/L4rralde/ReinforcementLearning/blob/main/docs/imgs/qlearning_algo.jpg" width="600"/>
</div>

For this algortihm, I developed the *SarsamaxQLearning* class within *qlearn.py* code. *qlearn_gym_train.py* and *qlearn_gym_demo.py* are working scripts for trainning and evaluate a quality table over OpenAI´s gym library. Let's dive deeper into these scripts:

&nbsp;
*qlearn_gym_train.py*
```python
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

```

&nbsp;
*qlearn_gym_demo.py*

```python
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
```

Voila, you may have a Q-table capable of playing *Taxi*
