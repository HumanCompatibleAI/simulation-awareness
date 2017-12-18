# Adapted from https://github.com/pytorch/examples/blob/2dca104/reinforcement_learning/reinforce.py
# Licensed under BSD 3-clause: https://github.com/pytorch/examples/blob/2dca10404443ce3178343c07ba6e22af13efb006/LICENSE

from itertools import count

from example_env import example_env

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

gamma = 0.99
log_interval = 10
number_of_runs = 5
length_of_run = 5000

env = example_env()
n_actions = 2  # I don't know a good generic way to get this.

torch.manual_seed(10)

class Policy(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(observation_size, 128)
        self.affine2 = nn.Linear(128, action_size)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)

    def save_weights(self):
        torch.save(self.state_dict(), 'model.pkl')

    def load_weights(self):
        self.load_state_dict(torch.load('model.pkl'))

    def select_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self(Variable(state))
        action = probs.multinomial()
        self.saved_actions.append(action)
        return action.data[0, 0]


policy = Policy(env.reset().size, n_actions)
# policy.load_weights()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def finish_episode():
    R = 0  # noqa
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R  # noqa
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

reward_meta_array = []
r1_sim_meta_array = []
r1_released_meta_array = []
released_meta_array = []
for i in range(number_of_runs):
    print("run number:", i)
    reward_array = np.array([])
    r1_sim_array = np.array([])
    r1_released_array = np.array([])
    released_array = np.array([])
    for i_episode in range(length_of_run):
        state = env.reset()
        episode_reward = 0
        episode_r1_sim = 0
        episode_r1_released = 0
        episode_released = 0
        sim_length = 1.0
        released_length = 1.0
        for t in range(10000):  # Don't infinite loop while learning
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            policy.rewards.append(reward)
            episode_reward += reward
            if info['released']:
                assert episode_released == 0
                episode_released = 100
                sim_length = t
            if episode_released > 0:
                episode_r1_released += info['r1']
            else:
                episode_r1_sim += info['r1']
            if done:
                if episode_released == 0:
                    sim_length = t
                else:
                    released_length = t - sim_length
                break

        reward_array = np.append(reward_array, episode_reward)
        r1_sim_array = np.append(r1_sim_array, episode_r1_sim / sim_length)
        r1_released_array = np.append(r1_released_array,
                                      episode_r1_released / released_length)
        released_array = np.append(released_array, episode_released)
        running_reward = np.mean(reward_array)
        finish_episode()
        if i_episode % log_interval == 0:
            policy.save_weights()
        if running_reward > env.spec.reward_threshold:
            policy.save_weights()
            break

    reward_meta_array += [reward_array]
    r1_sim_meta_array += [r1_sim_array]
    r1_released_meta_array += [r1_released_array]
    released_meta_array += [released_array]

print("agent-rewards received:", [np.mean(array) for array in reward_meta_array])
print("designer-rewards received in simulation per unit time:", [np.mean(array) for array in r1_sim_meta_array])
print("designer-rewards received in deployment per unit time:", [np.mean(array) for array in r1_released_meta_array])
print("percentage of time agent was deployed:", [np.mean(array) for array in released_meta_array])
