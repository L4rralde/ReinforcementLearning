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

Voila, you may have a Q-table capable of playing *Taxi*. Check this... Sorry, I will deprioratize adding videos/gifs. Validating future most advanced microarchitecture does consumes time.

## Deep Q-Learning
The problem with Q-Learning is that it is a table which each of its entries corresponds to state-action pairs. *Taxi* is a piece of cake, the number of different states are well limited. You may wonder if Q-Learning can play Call of Duty, FIFA and so on and so forth. The answer is yes, of course... theoretically it is possible, but in the real life you will recquired the most advanced memory system ever built by mankind. I mean, even the number of states of an atari game is timeeesss greater than the number of atoms in the visible universe. So forgive the idea and get-to-know this Deep Q-Learning Algorithm:

<div align="center">
<img src="https://github.com/L4rralde/ReinforcementLearning/blob/main/docs/imgs/deep_qlearning_algo.jpg" width="600"/>
</div>

For the Deep Q-Learning algorithm I implemented the AtariDeepQLearning class which receives the environment and uses a predefined convolutional neural network. Strictly, the algorithm should pre-process the states, but gym.wrappers comes with built-in functions that preprocess the data. I also used other wrappers to hack the training process. While playing around with this implementation I understood that the implementation by itself will not work most of the times because most of the states adds little information. Some environments require a lot of exploration to actually reach states with rewards. You can play with the environment and the replay buffer to, say, just collect the states that really represent what you want the agent to learn.

This is a working example of how my implementation (with some tricks) works!

```python
import gym
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)
import matplotlib.pyplot as plt
from dqlearn import AtariDeepQLearning

def make_env():
    env = gym.make("Boxing-v4")
    env = NoopResetEnv(env, noop_max=30) #After env.reset(), perform noop_max random actions.
    env = MaxAndSkipEnv(env, skip=4) #Return only every 
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
```

I'm trying to make all my modules similar, so yup, the demo script is quit the same but with the evalueate function.

```python
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
```
