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


def linear_schedule(episode: int, duration: float, min_value: float=0.01):
    slope = (1.0-min_value)/duration
    return max(min_value, 1.0+slope*episode)


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
            episodes: int = 10,
            timesteps: int = 10000,
            alpha: float = 1e-4,
            gamma: float = 0.99,
            decay_rate: float = 0.0001) -> None:
        replay_buffer = ReplayBuffer(self.env, 1000)

        #target_network = copy.deepcopy(self.quality_network) Does this really work?
        target_network = AtariDeepQNetwork(self.env).to(device)
        target_network.load_state_dict(self.quality_network.state_dict())

        optimizer = optim.Adam(self.quality_network.parameters(), lr=alpha)
        for episode in range(episodes):
            # Initialize sequence?
            state, _ = self.env.reset()
            losses = []
            rewards = []
            for timestep in tqdm(range(timesteps)):
                ## Sampling
                # greedy update policy
                epsilon = greedy_epsilon(episode*timesteps+timestep, decay_rate) #@L4rralde [FIXME]. how much should decay?
                action = self.update_policy(state, epsilon)
                # Execute action in emulator and observe
                new_state, reward, done, truncated, info = self.env.step(action)
                rewards.append(reward)
                # Store transition
                #@L4rralde [TODO]: How to handle final observation?
                replay_buffer.add(state, new_state, action, reward, done, info)
                state = new_state
                if (len(replay_buffer) < 500) or (len(replay_buffer)%4 != 0):
                    continue
                ##Training
                mini_batch = replay_buffer.sample(32)
                with torch.no_grad():
                    #@L4rralde. This is horrible
                    target_max, _ = target_network(mini_batch.next_states).max(dim=1)
                    td_target = mini_batch.rewards.flatten() + gamma*target_max*(1-mini_batch.dones.flatten())
                prediction = self.quality_network(mini_batch.states).gather(1, mini_batch.actions.type(torch.int64)).squeeze()
                loss = F.mse_loss(td_target, prediction)
                losses.append(loss)

                #Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Update target network every C steps
                if timestep%1000 == 0:
                    #target_network = copy.deepcopy(self.quality_network) #FIXME: is this efficient? I don't even know if deepcopy works across devices
                    #@L4rralde [TBD], should I use an update ratio, like target_param <- ratio*target_param + (1-ratio)*target_param
                    target_network.load_state_dict(self.quality_network.state_dict())
            if not(losses):
                print(f"episode {episode+1}/{episodes}. loss=NAN")
            else:
                print(f"episode {episode+1}/{episodes}. loss={sum(losses)/len(losses)}, rewards={sum(rewards)/len(rewards)}")

    def evaluate(self,
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

    def load(self, fname) -> None:
        self.quality_network.load_state_dict(torch.load(fname))

    def save(self, fname) -> None:
        torch.save(self.quality_network.state_dict(), fname)
