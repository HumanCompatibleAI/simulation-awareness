# Adapted from https://github.com/pytorch/examples/blob/2dca104/reinforcement_learning/reinforce.py
# Licensed under BSD 3-clause: https://github.com/pytorch/examples/blob/2dca10404443ce3178343c07ba6e22af13efb006/LICENSE

from itertools import count

from example_env import example_env

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

gamma = 0.99
log_interval = 10

env = example_env()
n_actions = 2  # I don't know a good generic way to get this.


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


running_reward = None
running_r1 = None
running_releases = 0
for i_episode in count(1):
    state = env.reset()
    episode_reward = 0
    episode_r1 = 0
    episode_released = 0
    for t in range(10000):  # Don't infinite loop while learning
        action = policy.select_action(state)
        state, reward, done, info = env.step(action)
        policy.rewards.append(reward)
        episode_reward += reward
        if info['released']:
            assert episode_released == 0
            episode_released = 100
        episode_r1 += info['r1']
        if done:
            break

    if running_reward is None:
        running_reward = episode_reward
    running_reward = running_reward * 0.99 + episode_reward * 0.01
    if running_r1 is None:
        running_r1 = episode_r1
    running_r1 = running_r1 * 0.99 + episode_r1 * 0.01
    running_releases = running_releases * 0.99 + episode_released * 0.01
    finish_episode()
    if i_episode % log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tAverage release %: {:.2f}%\tAverage R1 achieved: {:.2f}'.format(
            i_episode, episode_reward, running_reward, running_releases, running_r1))
        policy.save_weights()
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {:.2f} and "
              "the last episode received {:.2f} reward!".format(running_reward, episode_reward))
        policy.save_weights()
        break
