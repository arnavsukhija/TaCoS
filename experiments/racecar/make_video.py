import pickle
import jax.tree_util as jtu
import os
import matplotlib.pyplot as plt
from wtc.envs.rccar import RCCar, plot_rc_trajectory

def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'mjx',
               filename: str = None,
               track: bool = False,
               dir: str = 'random',
               plot: bool = False,
               ):
    assert env_name in ['rccar']
    assert backend in ['generalized', 'positional', 'spring', 'mjx']
    env = RCCar(margin_factor=20)

    with open(os.path.join(dir, filename), 'rb') as fp:
        trajectory = pickle.load(fp)

    fig, axs = plot_rc_trajectory(trajectory.obs)
    print('Done')


if __name__ == '__main__':
    environments = ['rccar']
    tracks = [True]
    for env, track in zip(environments[:1], tracks[:1]):
        experiment(env_name=env,
                   backend='generalized',
                   filename=f'trajectory_20Mil.pkl',
                   track=track,
                   dir='trajectories/PPO/newParams',
                   plot=True)
