import argparse
import datetime
import os
import pickle
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax import envs
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import ConstantSwitchCost, IHSwitchCostWrapper

ENTITY = 'trevenl'


def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'generalized',
               project_name: str = 'GPUSpeedTest',
               num_timesteps: int = 1_000_000,
               episode_length: int = 200,
               learning_discount_factor: int = 0.99,
               switch_cost: float = 0.1,
               min_reps: int = 1,
               max_reps: int = 30,
               seed: int = 0,
               num_envs: int = 32,
               num_env_steps_between_updates: int = 10,
               networks: int = 0,
               ):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    assert backend in ['generalized', 'positional', 'spring']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    if networks == 0:
        policy_hidden_layer_sizes = (32,) * 5
        critic_hidden_layer_sizes = (128,) * 4

    else:
        policy_hidden_layer_sizes = (64, 64)
        critic_hidden_layer_sizes = (64, 64)

    config = dict(env_name=env_name,
                  backend=backend,
                  episode_length=episode_length,
                  learning_discount_factor=learning_discount_factor,
                  switch_cost=switch_cost,
                  min_reps=min_reps,
                  max_reps=max_reps,
                  seed=seed,
                  num_envs=num_envs,
                  num_env_steps_between_updates=num_env_steps_between_updates,
                  networks=networks)

    wandb.init(
        project=project_name,
        dir='/cluster/scratch/' + ENTITY,
        config=config,
    )

    continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=learning_discount_factor,
                                                                dt=env.dt)

    switch_cost = ConstantSwitchCost(value=jnp.array(switch_cost))
    env = IHSwitchCostWrapper(env=env,
                              num_integrator_steps=episode_length,
                              min_time_between_switches=min_reps * env.dt,
                              max_time_between_switches=max_reps * env.dt,
                              switch_cost=switch_cost,
                              discounting=learning_discount_factor,
                              time_as_part_of_state=True,
                              )

    optimizer = SAC(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        action_repeat=1,
        num_env_steps_between_updates=num_env_steps_between_updates,
        num_envs=num_envs,
        num_eval_envs=4,
        lr_alpha=3e-4,
        lr_policy=3e-4,
        lr_q=3e-4,
        wd_alpha=0.,
        wd_policy=0.,
        wd_q=0.,
        max_grad_norm=1e5,
        discounting=learning_discount_factor,
        batch_size=32,
        num_evals=20,
        normalize_observations=True,
        reward_scaling=1.,
        tau=0.005,
        min_replay_size=10 ** 2,
        max_replay_size=10 ** 5,
        grad_updates_per_step=num_env_steps_between_updates * num_envs,
        deterministic_eval=True,
        init_log_alpha=0.,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        policy_activation=swish,
        critic_hidden_layer_sizes=critic_hidden_layer_sizes,
        critic_activation=swish,
        wandb_logging=True,
        return_best_model=True,
        non_equidistant_time=True,
        continuous_discounting=continuous_discounting,
        min_time_between_switches=min_reps * env.dt,
        max_time_between_switches=max_reps * env.dt,
    )

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
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(seed), progress_fn=progress)
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)

    @jax.jit
    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    ########################## Evaluation ##########################
    ################################################################

    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    env = IHSwitchCostWrapper(env=env,
                              num_integrator_steps=episode_length,
                              min_time_between_switches=min_reps * env.dt,
                              max_time_between_switches=max_reps * env.dt,
                              switch_cost=switch_cost,
                              discounting=1.0,
                              time_as_part_of_state=True, )

    state = env.reset(rng=jr.PRNGKey(0))

    def step(state, _):
        u = policy(state.obs)[0]
        next_state, rest = env.simulation_step(state, u)
        return next_state, (next_state.obs, u, next_state.reward, rest)

    init_state = state
    LEGEND_SIZE = 20
    LABEL_SIZE = 20
    TICKS_SIZE = 20

    import matplotlib as mpl

    mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
    mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

    trajectory = []
    full_trajectories = []
    while not state.done:
        state, one_traj = step(state, None)
        one_traj, full_trajectory = one_traj[:-1], one_traj[-1]
        trajectory.append(one_traj)
        full_trajectories.append(full_trajectory)

    trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
    full_trajectory = jtu.tree_map(lambda *xs: jnp.concatenate(xs), *full_trajectories)

    # We save full_trajectory to wandb
    # Save trajectory rather than rendered video
    directory = os.path.join(wandb.run.dir, 'results')
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, 'trajectory_1.pkl')
    with open(model_path, 'wb') as handle:
        pickle.dump(full_trajectory, handle)
    wandb.save(model_path, wandb.run.dir)

    xs_full_trajectory = jnp.concatenate([init_state.obs[:-1].reshape(1, -1), full_trajectory.obs, ])
    rewards_full_trajectory = jnp.concatenate([init_state.reward.reshape(1, ), full_trajectory.reward])
    executed_integration_steps = xs_full_trajectory.shape[0]

    ts_full_trajectory = env.env.dt * jnp.array(list(range(executed_integration_steps)))
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
    us = trajectory[1][:, :-1]
    times = trajectory[0][:, -1]

    # All times are the times when we ended the actions
    all_ts = times
    all_ts = jnp.concatenate([jnp.array([0.0]), all_ts])

    for i in range(xs_full_trajectory.shape[1]):
        axs[0].plot(ts_full_trajectory, xs_full_trajectory[:, i])
    for h in all_ts[:-1]:
        axs[0].axvline(x=h, color='black', ls='--', alpha=0.4)

    axs[0].set_xlabel('Time', fontsize=LABEL_SIZE)
    axs[0].set_ylabel('State', fontsize=LABEL_SIZE)

    axs[1].step(all_ts, jnp.concatenate([us, us[-1].reshape(1, -1)]), where='post', label=r'$u$')
    axs[1].set_xlabel('Time', fontsize=LABEL_SIZE)
    axs[1].set_ylabel('Action', fontsize=LABEL_SIZE)

    axs[2].plot(ts_full_trajectory, rewards_full_trajectory, label='Rewards')
    for h in all_ts[:-1]:
        axs[2].axvline(x=h, color='black', ls='--', alpha=0.4)

    axs[2].set_xlabel('Time', fontsize=LABEL_SIZE)
    axs[2].set_ylabel('Instance reward', fontsize=LABEL_SIZE)

    axs[3].plot(jnp.diff(all_ts), label='Times for actions')
    axs[3].set_xlabel('Action Steps', fontsize=LABEL_SIZE)
    axs[3].set_ylabel('Time for action', fontsize=LABEL_SIZE)

    for ax in axs:
        ax.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()

    wandb.log({'pendulum_switch_bound': wandb.Image(fig),
               'results/total_reward': float(jnp.sum(trajectory[2])),
               'results/num_actions': trajectory[0].shape[0]})

    wandb.finish()


def main(args):
    experiment(env_name=args.env_name,
               backend=args.backend,
               project_name=args.project_name,
               num_timesteps=args.num_timesteps,
               episode_length=args.episode_length,
               switch_cost=args.switch_cost,
               learning_discount_factor=args.learning_discount_factor,
               min_reps=args.min_reps,
               max_reps=args.max_reps,
               seed=args.seed,
               num_envs=args.num_envs,
               num_env_steps_between_updates=args.num_env_steps_between_updates,
               networks=args.networks
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hopper')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='GPUSpeedTest')
    parser.add_argument('--num_timesteps', type=int, default=100_000)
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--switch_cost', type=float, default=1.0)
    parser.add_argument('--learning_discount_factor', type=float, default=0.99)
    parser.add_argument('--min_reps', type=int, default=1)
    parser.add_argument('--max_reps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--num_env_steps_between_updates', type=int, default=10)
    parser.add_argument('--networks', type=int, default=1)

    args = parser.parse_args()
    main(args)
