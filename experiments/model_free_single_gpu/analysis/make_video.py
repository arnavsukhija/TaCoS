from brax import envs
import imageio
import pickle


def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'generalized',
               filename: str = None,
               track: bool = False
               ):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    assert backend in ['generalized', 'positional', 'spring']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    with open(filename, 'rb') as fp:
         trajectory = pickle.load(fp)

    if track:
        video_frames = env.render([s.pipeline_state for s in trajectory], camera='track')
    else:
        video_frames = env.render([s.pipeline_state for s in trajectory])

    with imageio.get_writer(f'video/{env_name}_video.mp4', fps=int(1 / env.dt)) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    print('Done')


if __name__ == '__main__':
    environments = ['halfcheetah', 'humanoid', 'ant', 'pusher', 'walker2d', 'reacher', 'inverted_double_pendulum']
    tracks = [True, True, True, False, True, False, False]
    for env, track in zip(environments[:1], tracks[:1]):
        experiment(env_name=env,
                   backend='generalized',
                   filename=f'data/{env}_trajectory.pkl',
                   track=track)