from collections import deque
from typing import NamedTuple

# Scientific
import numpy as np
import matplotlib.pyplot as plt
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Reinforce algorithm or Monte Carlo Reinforce.
# Reinfoce belongs to a special class of policy-based reinforcement algorithms called
# Policy Gradient algorithms

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.n, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax()
        )

    def forward(self, x):
        x = nn.Flatten(x)
        return self.network(x)


class MonteCarloReinforce:
    def __init__(self, env) -> None:
        self.env = env
        self.policy = Policy(env)

    def act(self, state) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state).cpu() #Policy returns the probabilities to take each action
        m = Categorical(probs)
        action = m.sample() #Action sampled given probabilities
        return (action.item(), m.log_prob(action))

    def train(self,
              n_episodes: int,
              max_n_steps: int,
              alpha: float,
              gamma: float,
              ) -> None:
        optimizer = optim.Adam(self.policy.parameters(), lr=alpha)

        scores_dq = deque(maxlen=100)
        scores = []

        for episode in range(n_episodes):
            saved_log_probs = []
            rewards = []

            #ALGORITHM: Generate an episode following Policy
            state, _ = self.env.reset()

            for t in range(max_n_steps):
                action, log_prob = self.act(state)
                next_state, reward, done, truncated, info = self.env.step(action)

                saved_log_probs.append(log_prob)
                rewards.appedn(reward)
                if done:
                    break
                state = next_state
            
            scores_dq.append(sum(rewards))
            scores.append(sum(rewards))
            returns = deque(maxlen=max_n_steps)
            n_steps = len(rewards)
            
            #ALGORITHM: for t  from T-1 to 0:
            disc_reward = 0
            for t in reversed(range(n_steps)): #TBD. Is this dynamic program correct??
                next_disc_reward = gamma*disc_reward + rewards[t]
                returns.appendleft(next_disc_reward)
                disc_reward = next_disc_reward

            #Standarize rewards to make training more stable
            eps = np.finfo(np.float32).eps.item()
            #eps is the smallest representation of float32, which is added to
            #the standard deviation of the returns to avoid numerical instabilities.
            returns = torch.tensor(returns)
            returns = (returns-returns.mean())/(returns.std()+eps)

            #ALGORITHM: Calculate loss
            loss = []
            for log_prob, disc_reward in zip(saved_log_probs, rewards):
                loss.append(-log_prob*disc_reward) #Negative sign is used because pytorch prefers gradient descend
            loss = torch.cat(loss).sum()
            #No need to average since returns was normalized (?)

            #ALGORITHM: Optimize Policy using gradient descend:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"episode={episode}/{n_episodes}. Avergae score={np.mean(scores_dq):.2f}")

    def evaluate(self, seeds: list) -> None:
        pass

    def save(self, fname: str) -> None:
        pass

    def load(self, fname: str) -> None:
        pass
