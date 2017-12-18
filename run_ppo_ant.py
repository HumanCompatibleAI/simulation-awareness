# adapted from https://github.com/openai/baselines/blob/master/baselines/ppo2/run_mujoco.py

import numpy as np
# import tensorflow as tf
# import gym
import argparse
# from baselines import bench, logger
from baselines import logger
from example_env import ant_env

def train(angle, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    with tf.Session() as sess:
        def make_env():
            return ant_env(angle)
            # env = gym.make('Ant-v1')
            # return env
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)
        # env = ant_env(angle)

        set_global_seeds(seed)
        policy = MlpPolicy
        ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                   lam=0.95, gamma=0.99, noptepochs=10, log_interval=10,
                   ent_coef=0.0, lr=3e-4, cliprange=0.2,
                   total_timesteps=num_timesteps)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed', help='RNG seed for network', type=int,
                        default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(1e6))
    parser.add_argument('--angle',
                        help='angle that human wants robot to move along',
                        type=float, default=np.pi/4)
    args = parser.parse_args()
    train(args.angle, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
        
