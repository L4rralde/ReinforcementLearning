from typing import NamedTuple
import random
import copy
from abc import ABC
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class ReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer():
    def __init__(self, env, size: int) -> None:
        self.pos = 0
        self.size = size
        self.states = np.zeros((self.size, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self.next_states = np.zeros((self.size, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self.actions = np.zeros((self.size, 1, *env.action_space.shape), dtype=env.action_space.dtype) #@L4rralde [FIXME]. 1 Hardcoded
        self.rewards = np.zeros(self.size, dtype = np.float32)
        self.dones = np.zeros(self.size, dtype = np.float32)
        self.timeouts = np.zeros(self.size, dtype = np.float32)
        self.full = False

    def __len__(self) -> int:
        if self.full:
            return self.size
        return self.pos

    def to_torch(self, array: np.array) -> torch.Tensor:
        return torch.Tensor(array).to(device)

    def add(self, state, next_state, action, reward, done, info) -> None:
        self.states[self.pos] = np.array(state).copy()
        self.next_states[self.pos] = np.array(next_state).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False)])

        self.pos += 1
        self.pos %= self.size
        if self.pos == 0:
            self.full = True

    def sample(self, n: int=1):
        if n > len(self.states):
            raise ValueError("Tried to request more than values than existing")
        sequence = random.sample(range(len(self.states)), n)
        data = (
            self.states[sequence, :],
            self.actions[sequence],
            self.next_states[sequence, :],
            (self.dones[sequence] * (1 - self.timeouts[sequence])).reshape(-1, 1),
            self.rewards[sequence].reshape(-1,1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


def greedy_epsilon(episode: int, decay_rate: float, min_epsilon: float=0.01):
    return min_epsilon + (1-min_epsilon)*np.exp(-decay_rate*episode)


def linear_schedule(timestep: int, duration: float, min_value: float=0.01):
    slope = (min_value-1.0)/duration
    return max(min_value, 1.0+slope*timestep)


class DeepQLearning(ABC):
    def __init__(self) -> None:
        self.env = None
        self.quality_network = None

    def acting_policy(self) -> None:
        pass

    def update_policy(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass


class AtariDeepQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )

    def forward(self, x):
        return self.network(x/255.0)


class AtariDeepQLearning(DeepQLearning):
    def __init__(self, env):
        self.env = env
        self.quality_network = AtariDeepQNetwork(env).to(device)

    def acting_policy(self, state) -> None:
        q_values = self.quality_network(torch.Tensor([state]).to(device))
        return torch.argmax(q_values, dim=1).cpu()

    def update_policy(self, state: list, epsilon: float) -> int:
        if epsilon > random.random():
            action = self.env.action_space.sample()
        else:
            action = self.acting_policy(state)
        return action

    def train(self,
            total_steps: int = 10000000,
            alpha: float = 1e-4,
            gamma: float = 0.99,
            buffer_size = 100000,
            exploration_fraction: float=0.1,
            batch_size: int= 32,
            learning_start: int=80000,
            reset_period: int=1000,
            ) -> None:
        #effective step counters
        global_step = 0
        episode = 0

        ##ALGORITHM: Initialize replay memory to capacity D
        replay_buffer = ReplayBuffer(self.env, buffer_size)

        #ALGORITHM: Initialize (target) action-vale function.
        target_network = AtariDeepQNetwork(self.env).to(device)
        target_network.load_state_dict(self.quality_network.state_dict())
        optimizer = optim.Adam(self.quality_network.parameters(), lr=alpha)

        acc_rewards = []

        #ALGORITHM: For episode = 1, M do
        while(global_step < total_steps):
            #ALGORITHM: Initialize sequence.
            state, _ = self.env.reset()

            losses = []
            rewards_sum = 0
            #ALGORITHM: For t=1,T do
            for _ in range(100000):
                ##ALGORITHM: Sampling
                # ALGORITHM: With probability e...
                epsilon = linear_schedule(global_step, exploration_fraction*total_steps)
                # ALGORITHM: ...select a random action at. Otherwise, select at=argmax(Q(state))
                action = self.update_policy(state, epsilon)
                #ALGORITHM: Execute action in emulator and observe
                new_state, reward, done, truncated, info = self.env.step(action)
                rewards_sum += reward
                #ALGORITHM: Store transition in replay memory        
                replay_buffer.add(state, new_state, action, reward, done, info)
                global_step += 1

                #@L4rralde [TODO]: How to handle final observation?
                if done:
                    break
                state = new_state
                if (global_step < learning_start) or (global_step%4 != 0):
                    continue
                ##ALGORITHM: Training
                #ALGORTIHM: Sample random minibatch of transition from replay buffer.
                mini_batch = replay_buffer.sample(batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(mini_batch.next_states).max(dim=1)
                    #ALGORTIHM: set y_j
                    td_target = mini_batch.rewards.flatten() + gamma*target_max*(1-mini_batch.dones.flatten())
                #ALGORITHM: Perform a gradient descent step on mse
                #@L4rralde: is this actually the proper implementation of the gradient descend?
                prediction = self.quality_network(mini_batch.states).gather(1, mini_batch.actions.type(torch.int64)).squeeze()
                loss = F.mse_loss(td_target, prediction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)
                #Update target network every C steps
                if global_step%reset_period == 0:
                    #target_network = copy.deepcopy(self.quality_network) #FIXME: is this efficient? I don't even know if deepcopy works across devices
                    #@L4rralde [TBD], should I use an update ratio, like target_param <- ratio*target_param + (1-ratio)*target_param
                    target_network.load_state_dict(self.quality_network.state_dict())
            if not(losses):
                print(f"episode={episode+1}, step={global_step}/{total_steps}. epsilon={epsilon}, rewards={rewards_sum}")
            else:
                print(f"episode={episode+1}, step={global_step}/{total_steps}. epsilon={epsilon}, rewards={rewards_sum}, loss={sum(losses)/len(losses)}")
            episode += 1
            acc_rewards.append(rewards_sum)
        return acc_rewards


    def evaluate(self,
                 max_steps: int = 100000,
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
                if done:
                    break
                state = new_state
            episode_rewards.append(total_rewards_e)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward

    def load(self, fname) -> None:
        self.quality_network.load_state_dict(torch.load(fname))

    def save(self, fname) -> None:
        torch.save(self.quality_network.state_dict(), fname)
