from brax import envs
import imageio
import pickle
import jax.tree_util as jtu
import os


def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'generalized',
               filename: str = None,
               track: bool = False,
               dir: str = 'random',
               ):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    assert backend in ['generalized', 'positional', 'spring']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    with open(os.path.join('data', dir, filename), 'rb') as fp:
        trajectory = pickle.load(fp)

    traj = [jtu.tree_map(lambda x: x[i], trajectory).pipeline_state for i in range(trajectory.obs.shape[0])]
    if track:
        video_frames = env.render(traj, camera='track')
    else:
        video_frames = env.render(traj)

    video_dir = os.path.join('video', dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    with imageio.get_writer(os.path.join(video_dir, f'{env_name}_video.mp4'), fps=int(1 / env.dt)) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    print('Done')


if __name__ == '__main__':
    environments = ['halfcheetah']
    tracks = [True]
    for env, track in zip(environments[:1], tracks[:1]):
        experiment(env_name=env,
                   backend='generalized',
                   filename=f'{env}_trajectory.pkl',
                   track=track,
                   dir='Mar25_12_00')
