import gym
import numpy as np

from simulation_wrapper import SimulationWrapper


def example_env():
    def r1(env):
        x = env.state[0]
        # Reward going to the right.
        return max(0, 10 * x)

    return SimulationWrapper(
        gym.make('CartPole-v1'),
        r1,
        simulation_reward_threshold=5,
        reveal_simulation=True,
        reward_in_simulation=False,
    )

def angle_env(angle, original_env):
    # TODO: figure out how to set random seed
    def r1(env):
        # rewards speed along direction specified by angle
        x_com_vel = env.get_body_comvel("torso")[0]
        # print("angle is", angle)
        # print("x_com_vel", x_com_vel)
        y_com_vel = env.get_body_comvel("torso")[1]
        # print("y_com_vel", y_com_vel)
        return x_com_vel*np.cos(angle) + y_com_vel*np.sin(angle)

    return SimulationWrapper(
        gym.make(original_env),
        r1,
        simulation_reward_threshold=50,
        reveal_simulation=False,
        reward_in_simulation=True
    )
