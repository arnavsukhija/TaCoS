import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import envs

ENTITY = 'trevenl'

env_name = 'inverted_pendulum'
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

# Do 10 base steps
num_divisors = 1
state = init_state
all_states = [init_state.obs]
env.sys = env.sys.replace(dt=base_dt / num_divisors)
jitted_step = jax.jit(env.step)
for i in range(num_steps * num_divisors):
    state = jitted_step(state, action)
    all_states.append(state.obs)
all_states = jnp.stack(all_states)

plt.plot(all_states)
plt.title(f'Number of divisors: {num_divisors}')
plt.show()


# Do 10 base steps
num_divisors = 10
state = init_state
all_states = [init_state.obs]
env.sys = env.sys.replace(dt=base_dt / num_divisors)
jitted_step = jax.jit(env.step)
for i in range(num_steps * num_divisors):
    state = jitted_step(state, action)
    all_states.append(state.obs)
all_states = jnp.stack(all_states)

plt.plot(all_states)
plt.title(f'Number of divisors: {num_divisors}')
plt.show()

# optimizer = SAC(
#     environment=env,
#     num_timesteps=num_timesteps,
#     episode_length=episode_length,
#     action_repeat=action_repeat,
#     num_env_steps_between_updates=num_env_steps_between_updates,
#     num_envs=num_envs,
#     num_eval_envs=32,
#     lr_alpha=3e-4,
#     lr_policy=3e-4,
#     lr_q=3e-4,
#     wd_alpha=0.,
#     wd_policy=0.,
#     wd_q=0.,
#     max_grad_norm=1e5,
#     discounting=learning_discount_factor,
#     batch_size=batch_size,
#     num_evals=20,
#     normalize_observations=True,
#     reward_scaling=reward_scaling,
#     tau=0.005,
#     min_replay_size=10 ** 3,
#     max_replay_size=10 ** 6,
#     grad_updates_per_step=num_env_steps_between_updates * num_envs,
#     deterministic_eval=True,
#     init_log_alpha=0.,
#     policy_hidden_layer_sizes=policy_hidden_layer_sizes,
#     policy_activation=swish,
#     critic_hidden_layer_sizes=critic_hidden_layer_sizes,
#     critic_activation=swish,
#     wandb_logging=False,
#     return_best_model=True,
# )
#
# xdata, ydata = [], []
# times = [datetime.now()]
#
#
# def progress(num_steps, metrics):
#     times.append(datetime.now())
#     xdata.append(num_steps)
#     ydata.append(metrics['eval/episode_reward'])
#     plt.xlabel('# environment steps')
#     plt.ylabel('reward per episode')
#     plt.plot(xdata, ydata)
#     plt.show()
#
#
# print('Before inference')
# policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(seed), progress_fn=progress)
# print('After inference')
#
# # Now we plot the evolution
# pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)
#
#
# @jax.jit
# def policy(obs):
#     return pseudo_policy(obs, key_sample=jr.PRNGKey(0))
#
#
# ########################## Evaluation ##########################
# ################################################################
#
# env = envs.get_environment(env_name=env_name,
#                            backend=backend)
#
# state = env.reset(rng=jr.PRNGKey(0))
# step_fn = jax.jit(env.step)
#
# trajectory = []
# total_steps = 0
# while (not state.done) and (total_steps < episode_length):
#     action = policy(state.obs)[0]
#     for _ in range(action_repeat):
#         state = step_fn(state, action)
#         total_steps += 1
#         trajectory.append(state)
#
# trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
#
# plt.plot(trajectory.reward)
# plt.show()
