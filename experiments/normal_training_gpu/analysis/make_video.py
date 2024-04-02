from brax import envs
import imageio
import pickle
import jax.tree_util as jtu
import os
import matplotlib.pyplot as plt

def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'mjx',
               filename: str = None,
               track: bool = False,
               dir: str = 'random',
               plot: bool = False,
               ):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    assert backend in ['generalized', 'positional', 'spring', 'mjx']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    with open(os.path.join('data', dir, filename), 'rb') as fp:
        trajectory = pickle.load(fp)

    if plot:
        fig, axs = plt.subplots(ncols=1, nrows=1)
        # Plot trajectory:
        axs.plot(trajectory.obs)
        axs.set_ylabel('State')
        axs.set_xlabel('Steps')
        plt.show()

    traj = [jtu.tree_map(lambda x: x[i], trajectory).pipeline_state for i in range(trajectory.obs.shape[0])]
    if track:
        video_frames = env.render(traj, camera='track')
    else:
        video_frames = env.render(traj)

    video_dir = os.path.join('video', dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    new_filename = filename.replace('.pkl', '.mp4')

    with imageio.get_writer(os.path.join(video_dir, new_filename), fps=int(1 / env.dt)) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    print('Done')


if __name__ == '__main__':
    environments = ['humanoid']
    tracks = [True]
    for env, track in zip(environments[:1], tracks[:1]):
        for index in [1]:
            experiment(env_name=env,
                       backend='mjx',
                       filename=f'{env}_{index}.pkl',
                       track=track,
                       dir='Mar27_11_00',
                       plot=True)
