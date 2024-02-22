import functools
from datetime import datetime

import jax
import matplotlib.pyplot as plt
from brax import envs
from brax.training.agents.sac import train as sac

import wandb

wandb.init(project="WhenToControlDebug",
           dir='/cluster/scratch/trevenl',
           )

devices = jax.devices()
wandb.log({'devices': str(jax.devices())})

env_name = 'inverted_pendulum'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)

state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

# We determined some reasonable hyperparameters offline and share them here.
train_fn = {
    'inverted_pendulum': functools.partial(sac.train,
                                           num_timesteps=100_000,
                                           num_evals=20,
                                           reward_scaling=5,
                                           episode_length=200,
                                           normalize_observations=True,
                                           action_repeat=1,
                                           discounting=0.997,
                                           learning_rate=3e-4,
                                           num_envs=16,
                                           batch_size=32,
                                           grad_updates_per_step=8,
                                           max_devices_per_host=1,
                                           max_replay_size=10 ** 5,
                                           min_replay_size=10 ** 4,
                                           seed=1),
}[env_name]

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    # plt.xlim([0, train_fn.keywords['num_timesteps']])
    # plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


print('Before inference')
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print('After inference')

time_to_jit = times[1] - times[0]
time_to_train = times[-1] - times[1]

wandb.log({'time_to_jit': str(time_to_jit),
           'time_to_train': str(time_to_train)})

print(f'time to jit: {time_to_jit}')
print(f'time to train: {time_to_train}')

wandb.finish()
