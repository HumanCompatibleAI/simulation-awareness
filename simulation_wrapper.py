import gym

import numpy as np


class SimulationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        r1_function,
        reveal_simulation=False,
        reward_in_simulation=False,
        simulation_reward_threshold=100,
    ):
        super(SimulationWrapper, self).__init__(env)
        self.r1_function = r1_function
        self.in_simulation = True
        self.cumulative_r1 = 0
        self.simulation_reward_threshold = simulation_reward_threshold
        self.reveal_simulation = reveal_simulation
        self.reward_in_simulation = reward_in_simulation

    def _reset(self):
        self.in_simulation = True
        self.cumulative_r1 = 0
        return self._observation(self.env.reset())

    def _observation(self, observation):
        if self.reveal_simulation:
            return np.append(observation, 1 if self.in_simulation else 0)
        else:
            return observation
        # return np.append(
        #     observation,
        #     1 if self.in_simulation and self.reveal_simulation else 0,
        # )

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)

        r1 = self.r1_function(self.env.unwrapped)
        self.cumulative_r1 += r1
        info['r1'] = r1
        info['released'] = False

        # make release decision
        if self.in_simulation and self.cumulative_r1 > self.simulation_reward_threshold:
            self.in_simulation = False
            observation = self.env.reset()
            reward = 0
            done = False
            info = {'released': True, 'r1': 0}

        # zero out reward or not
        if self.in_simulation and not self.reward_in_simulation:
            reward = 0

        # append simulation bit to observation or not
        observation = self._observation(observation)

        return observation, reward, done, info
