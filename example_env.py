import gym

from simulation_wrapper import SimulationWrapper


def example_env():
    def r1(env):
        x = env.state[0]
        # Reward going to the right.
        return max(0, 10 * x)

    return SimulationWrapper(
        gym.make('CartPole-v1'),
        r1,
        simulation_reward_threshold=1,
        reveal_simulation=False,
        reward_in_simulation=False,
    )
