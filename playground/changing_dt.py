import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import envs

ENTITY = 'trevenl'

env_name = 'humanoid'
backend = 'generalized'
project_name = 'GPUSpeedTest'
num_timesteps = 1_000_000
episode_length = 200
learning_discount_factor = 0.99
seed = 0
num_envs = 32
num_env_steps_between_updates = 10
networks = 0
batch_size = 64
action_repeat = 1
reward_scaling = 1.0

assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                    'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d', 'drone', 'greenhouse']
env = envs.get_environment(env_name=env_name,
                           backend=backend)

if networks == 0:
    policy_hidden_layer_sizes = (32,) * 5
    critic_hidden_layer_sizes = (128,) * 4

else:
    policy_hidden_layer_sizes = (64, 64)
    critic_hidden_layer_sizes = (64, 64)

config = dict(env_name=env_name,
              num_timesteps=num_timesteps,
              episode_length=episode_length,
              learning_discount_factor=learning_discount_factor,
              seed=seed,
              num_envs=num_envs,
              num_env_steps_between_updates=num_env_steps_between_updates,
              networks=networks,
              batch_size=batch_size,
              action_repeat=action_repeat,
              reward_scaling=reward_scaling)

init_state = env.reset(rng=jax.random.PRNGKey(0))
action = 0.1 * jnp.ones(shape=(env.action_size,))
base_dt = env.sys.dt
num_steps = 10

# # Do 10 base steps
# num_divisors = 1
# state = init_state
# all_states = [init_state.obs]
# env.sys = env.sys.replace(dt=base_dt / num_divisors)
# jitted_step = jax.jit(env.step)
# for i in range(num_steps * num_divisors):
#     state = jitted_step(state, action)
#     all_states.append(state.obs)
# all_states = jnp.stack(all_states)
#
# plt.plot(all_states)
# plt.title(f'Number of divisors: {num_divisors}')
# plt.show()
#
# # Do 10 base steps
# num_divisors = 10
# state = init_state
# all_states = [init_state.obs]
# env.sys = env.sys.replace(dt=base_dt / num_divisors)
# jitted_step = jax.jit(env.step)
# for i in range(num_steps * num_divisors):
#     state = jitted_step(state, action)
#     all_states.append(state.obs)
# all_states = jnp.stack(all_states)
#
# plt.plot(all_states)
# plt.title(f'Number of divisors: {num_divisors}')
# plt.show()