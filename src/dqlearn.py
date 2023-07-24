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


class ReplayBuffer():
    def __init__(self, max_len, states: list=[], next_states: list=[], actions: list=[], rewards: list=[], dones: list=[], infos: list=[]) -> None:
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.infos = infos
        self.max_len = max_len
        self.cnt = 0

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int):
        return ReplayBuffer(self.max_len, [self.states[i]], [self.next_states[i]], [self.actions[i]], [self.rewards[i]], [self.dones[i]], [self.infos[i]])

    def add(self, state, next_state, action, reward, done, info) -> None:
        if self.cnt < len(self.states):
            self.states[self.cnt] = state
            self.next_states[self.cnt] = next_state
            self.actions[self.cnt] = action
            self.rewards[self.cnt] = reward
            self.dones[self.cnt] = done
            self.infos[self.cnt] = info
        else:
            self.states.append(state)
            self.next_states.append(next_state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.infos.append(info)
        self.cnt += 1
        self.cnt = self.cnt%self.max_len

    def sample(self, n: int=1):
        if n > len(self.states):
            raise ValueError("Tried to request more than values than existing")
        sequence = random.sample(range(len(self.states)), n)
        states = [self.states[i] for i in sequence]
        next_states = [self.next_states[i] for i in sequence]
        actions = [self.actions[i] for i in sequence]
        rewards = [self.rewards[i] for i in sequence]
        dones = [self.rewards[i] for i in sequence]
        infos = [self.infos[i] for i in sequence]
        return ReplayBuffer(self.max_len, states, next_states, actions, rewards, dones, infos)


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
            timesteps: int = 1000,
            alpha: float = 1e-4,
            gamma: float = 0.99,
            decay_rate: float = 0.0005) -> None:
        replay_buffer = ReplayBuffer(1000)

        #target_network = copy.deepcopy(self.quality_network) Does this really work?
        target_network = AtariDeepQNetwork(self.env).to(device)
        target_network.load_state_dict(self.quality_network.state_dict())

        optimizer = optim.Adam(self.quality_network.parameters(), lr=alpha)
        for episode in tqdm(range(episodes)):
            # Initialize sequence?
            state, _ = self.env.reset()
            for timestep in range(timesteps):
                ## Sampling
                # greedy update policy
                epsilon = greedy_epsilon(episode, decay_rate)
                action = self.update_policy(state, epsilon)
                # Execute action in emulator and observe
                new_state, reward, done, truncated, info = self.env.step(action)
                # Store transition
                replay_buffer.add(state, new_state, action, reward, done, info)
                if (len(replay_buffer) < 500) or (len(replay_buffer)%4 != 0):
                    continue
                ##Training
                mini_batch = replay_buffer.sample(32)
                with torch.no_grad():
                    #@L4rralde. This is horrible
                    target_max, _ = target_network(torch.Tensor(mini_batch.next_states).to(device)).max(dim=1)
                    td_target = torch.Tensor(mini_batch.rewards).to(device).flatten() + gamma*target_max*(1-torch.Tensor(mini_batch.dones).to(device).flatten())
                prediction = self.quality_network(torch.Tensor(mini_batch.states).to(device)).gather(1, torch.Tensor([mini_batch.actions]).to(device).type(torch.int64)).squeeze()
                loss = F.mse_loss(td_target, prediction)

                #Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Update target network every C steps
                if timestep%100 == 0:
                    #target_network = copy.deepcopy(self.quality_network) #FIXME: is this efficient? I don't even know if deepcopy works across devices
                    #@L4rralde [TBD], should I use an update ratio, like target_param <- ratio*target_param + (1-ratio)*target_param
                    target_network.load_state_dict(self.quality_network.state_dict())

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
