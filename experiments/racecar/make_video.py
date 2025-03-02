from brax import envs
import imageio
import pickle
import jax.tree_util as jtu
import os
import matplotlib.pyplot as plt
from wtc.envs.rccar import RCCar

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

    if plot:
        fig, axs = plt.subplots(ncols=1, nrows=1)
        # Plot trajectory:
        axs.plot(trajectory.obs)
        axs.set_ylabel('State')
        axs.set_xlabel('Steps')
        plt.show()

    traj = [jtu.tree_map(lambda x: x[i], trajectory).pipeline_state for i in range(trajectory.obs.shape[0])]
    # Get the number of steps in the trajectory
    steps = len(traj)

    # Call the render_from_trajectory function
    video_frames = env.render(traj)

    video_dir = os.path.join('video', dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    new_filename = filename.replace('.pkl', '.mp4')

    with imageio.get_writer(os.path.join(video_dir, new_filename), fps=int(10 / env.dt)) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    print('Done')


if __name__ == '__main__':
    environments = ['rccar']
    tracks = [True]
    for env, track in zip(environments[:1], tracks[:1]):
        for index in range(1):
            experiment(env_name=env,
                       backend='generalized',
                       filename=f'trajectory_{index}.pkl',
                       track=track,
                       dir='trajectories/TaCoS',
                       plot=True)
