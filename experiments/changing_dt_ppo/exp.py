import argparse
import datetime
import os
import pickle
from datetime import datetime
import math

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax import envs
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.ppo.ppo_brax_env import PPO

from wtc.wrappers.change_integration_dt import ChangeIntegrationStep
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import ConstantSwitchCost, IHSwitchCostWrapper
from wtc.envs.rccar import RCCar, plot_rc_trajectory
from wtc.envs.reacher_dm_control import ReacherDMControl

from jax import config

config.update("jax_debug_nans", True)

ENTITY = 'trevenl'


def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'generalized',
               project_name: str = 'GPUSpeedTest',
               num_timesteps: int = 1_000_000,
               episode_time: float = 5,
               base_dt_divisor: int = 1,
               base_discount_factor: int = 0.99,
               seed: int = 0,
               num_envs: int = 4096,
               num_eval_envs: int = 128,
               entropy_cost: int = 1e-2,
               unroll_length: int = 5,
               num_minibatches: int = 32,
               num_updates_per_batch: int = 4,
               batch_size: int = 64,
               networks: int = 0,
               reward_scaling: float = 1.0,
               switch_cost_wrapper: bool = False,
               switch_cost: float = 0.1,
               max_time_between_switches: float = 0.1,
               time_as_part_of_state: bool = True,
               num_final_evals: int = 10
               ):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d', 'drone', 'greenhouse', 'rccar']
    if env_name == 'rccar':
        # Episode time needs to be 4.0 seconds
        base_dt = 0.5
        base_episode_steps = 8
        new_dt = base_dt / base_dt_divisor
        env = RCCar(margin_factor=20, dt=new_dt)
    else:
        if env_name == 'reacher':
            env = ReacherDMControl(backend=backend)
        else:
            env = envs.get_environment(env_name=env_name,
                                       backend=backend)

        base_dt = env.dt
        base_episode_steps = episode_time // env.dt
        print(f'Base integration dt {base_dt}')
        print(f'Base episode steps: {base_episode_steps}')

        env = ChangeIntegrationStep(env=env,
                                    dt_divisor=base_dt_divisor)

    print(f'New integration dt {env.dt}')
    print(f'New episode steps: {episode_time // env.dt}')

    new_discount_factor = base_discount_factor ** (1 / base_dt_divisor)

    if switch_cost_wrapper:
        continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=new_discount_factor,
                                                                    dt=env.dt)

        env = IHSwitchCostWrapper(env=env,
                                  num_integrator_steps=int(episode_time // env.dt),
                                  min_time_between_switches=1 * env.dt,  # Hardcoded to be at least the integration step
                                  max_time_between_switches=max_time_between_switches,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(switch_cost)),
                                  discounting=new_discount_factor,
                                  time_as_part_of_state=time_as_part_of_state,
                                  )

    if networks == 0:
        policy_hidden_layer_sizes = (32,) * 5
        critic_hidden_layer_sizes = (128,) * 4

    else:
        policy_hidden_layer_sizes = (64, 64)
        critic_hidden_layer_sizes = (64, 64)

    config = dict(env_name=env_name,
                  backend=backend,
                  num_timesteps=num_timesteps,
                  episode_time=episode_time,
                  base_integration_dt=base_dt,
                  base_episode_steps=base_episode_steps,
                  new_integration_dt=env.dt,
                  new_episode_steps=episode_time // env.dt,
                  base_discount_factor=base_discount_factor,
                  base_dt_divisor=base_dt_divisor,
                  new_discount_factor=new_discount_factor,
                  seed=seed,
                  num_envs=num_envs,
                  num_eval_envs=num_eval_envs,
                  entropy_cost=entropy_cost,
                  unroll_length=unroll_length,
                  num_minibatches=num_minibatches,
                  num_updates_per_batch=num_updates_per_batch,
                  networks=networks,
                  batch_size=batch_size,
                  reward_scaling=reward_scaling,
                  switch_cost_wrapper=switch_cost_wrapper,
                  switch_cost=switch_cost,
                  max_time_between_switches=max_time_between_switches,
                  time_as_part_of_state=time_as_part_of_state,
                  num_final_evals=num_final_evals
                  )

    wandb.init(
        project=project_name,
        dir='/cluster/scratch/' + ENTITY,
        config=config,
    )

    if switch_cost_wrapper:
        optimizer = PPO(
            environment=env,
            num_timesteps=num_timesteps,
            episode_length=int(episode_time // env.dt),
            action_repeat=1,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=3e-4,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=new_discount_factor,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            num_evals=20,
            normalize_observations=True,
            reward_scaling=reward_scaling,
            max_grad_norm=1e5,
            clipping_epsilon=0.3,
            gae_lambda=0.95,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            policy_activation=swish,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_activation=swish,
            deterministic_eval=True,
            normalize_advantage=True,
            wandb_logging=True,
            return_best_model=True,
            non_equidistant_time=True,
            continuous_discounting=continuous_discounting,
            min_time_between_switches=1 * env.dt,
            max_time_between_switches=max_time_between_switches,
            env_dt=env.dt,
        )
    else:
        optimizer = PPO(
            environment=env,
            num_timesteps=num_timesteps,
            episode_length=int(episode_time // env.dt),
            action_repeat=1,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=3e-4,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=discrete_to_continuous_discounting(),
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            num_evals=20,
            normalize_observations=True,
            reward_scaling=reward_scaling,
            max_grad_norm=1e5,
            clipping_epsilon=0.3,
            gae_lambda=0.95,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            policy_activation=swish,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_activation=swish,
            deterministic_eval=True,
            normalize_advantage=True,
            wandb_logging=True,
            return_best_model=True,
        )

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
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

    print(f'Starting with evaluation')
    if switch_cost_wrapper:
        if env_name == 'rccar':
            # Episode time needs to be 4.0 seconds
            base_dt = 0.5
            new_dt = base_dt / base_dt_divisor
            env = RCCar(margin_factor=20, dt=new_dt)
        else:
            if env_name == 'reacher':
                env = ReacherDMControl(backend=backend)
            else:
                env = envs.get_environment(env_name=env_name,
                                           backend=backend)
            env = ChangeIntegrationStep(env=env,
                                        dt_divisor=base_dt_divisor)

        env = IHSwitchCostWrapper(env=env,
                                  num_integrator_steps=int(episode_time // env.dt),
                                  min_time_between_switches=1 * env.dt,
                                  max_time_between_switches=max_time_between_switches,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  discounting=1.0,
                                  time_as_part_of_state=time_as_part_of_state, )

        for index in range(num_final_evals):
            state = env.reset(rng=jr.PRNGKey(index))
            print(f'Prepared and reseted environment')

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

            print('Starting with trajectory simulation')
            trajectory = []
            full_trajectories = []
            while not state.done:
                state, one_traj = step(state, None)
                one_traj, full_trajectory = one_traj[:-1], one_traj[-1]
                trajectory.append(one_traj)
                full_trajectories.append(full_trajectory)

            print('End of trajectory simulation')
            trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
            full_trajectory = jtu.tree_map(lambda *xs: jnp.concatenate(xs), *full_trajectories)

            wandb.log({f'results/total_reward_{index}': float(jnp.sum(trajectory[2])),
                       f'results/num_actions_{index}': trajectory[0].shape[0]})

            print('Saving the models to Wandb')
            # We save full_trajectory to wandb
            # Save trajectory rather than rendered video
            directory = os.path.join(wandb.run.dir, 'results')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join(directory, f'trajectory_{index}.pkl')
            with open(model_path, 'wb') as handle:
                pickle.dump(full_trajectory, handle)
            wandb.save(model_path, wandb.run.dir)
            print('Trajectory saved to Wandb')

        print('Started plotting')
        if time_as_part_of_state:
            xs_full_trajectory = jnp.concatenate([init_state.obs[:-1].reshape(1, -1), full_trajectory.obs, ])
        else:
            xs_full_trajectory = jnp.concatenate([init_state.obs.reshape(1, -1), full_trajectory.obs, ])
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

        print('End of plotting, uploading results to wandb')

        wandb.log({'switch_bound_figure': wandb.Image(fig), })

        print('Results uploaded to wandb')

    else:
        if env_name == 'rccar':
            # Episode time needs to be 4.0 seconds
            base_dt = 0.5
            new_dt = base_dt / base_dt_divisor
            env = RCCar(margin_factor=20, dt=new_dt)
        else:
            if env_name == 'reacher':
                env = ReacherDMControl(backend=backend)
            else:
                env = envs.get_environment(env_name=env_name,
                                           backend=backend)
            env = ChangeIntegrationStep(env=env,
                                        dt_divisor=base_dt_divisor)

        step_fn = jax.jit(env.step)
        reset_fn = jax.jit(env.reset)
        for index in range(num_final_evals):
            state = reset_fn(rng=jr.PRNGKey(index))
            trajectory = []
            total_steps = 0
            while (not state.done) and (total_steps < (episode_time // env.dt)):
                action = policy(state.obs)[0]
                for _ in range(1):
                    state = step_fn(state, action)
                    total_steps += 1
                    trajectory.append(state)

            trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
            wandb.log({f'results/total_reward_{index}': jnp.sum(trajectory.reward),
                       f'results/num_actions_{index}': len(trajectory.reward)})

            print(f'Total reward {index}: {jnp.sum(trajectory.reward)}')
            print(f'Total steps {index}: {total_steps}')

            plt.plot(trajectory.reward)
            plt.show()

            # We save full_trajectory to wandb
            # Save trajectory rather than rendered video
            directory = os.path.join(wandb.run.dir, 'results')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join(directory, f'trajectory_{index}.pkl')
            with open(model_path, 'wb') as handle:
                pickle.dump(trajectory, handle)
            wandb.save(model_path, wandb.run.dir)

    wandb.finish()


def main(args):
    experiment(env_name=args.env_name,
               backend=args.backend,
               project_name=args.project_name,
               num_timesteps=args.num_timesteps,
               episode_time=args.episode_time,
               base_dt_divisor=args.base_dt_divisor,
               base_discount_factor=args.base_discount_factor,
               seed=args.seed,
               num_envs=args.num_envs,
               num_eval_envs=args.num_eval_envs,
               entropy_cost=args.entropy_cost,
               unroll_length=args.unroll_length,
               num_minibatches=args.num_minibatches,
               num_updates_per_batch=args.num_updates_per_batch,
               batch_size=args.batch_size,
               networks=args.networks,
               reward_scaling=args.reward_scaling,
               switch_cost_wrapper=bool(args.switch_cost_wrapper),
               switch_cost=args.switch_cost,
               max_time_between_switches=args.max_time_between_switches,
               time_as_part_of_state=bool(args.time_as_part_of_state),
               num_final_evals=args.num_final_evals
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='reacher')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='GPUSpeedTest')
    parser.add_argument('--num_timesteps', type=int, default=100_000)
    parser.add_argument('--episode_time', type=float, default=2.0)
    parser.add_argument('--base_dt_divisor', type=int, default=1)
    parser.add_argument('--base_discount_factor', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_eval_envs', type=int, default=64)
    parser.add_argument('--entropy_cost', type=float, default=5.0)
    parser.add_argument('--unroll_length', type=int, default=10)
    parser.add_argument('--num_minibatches', type=int, default=10)
    parser.add_argument('--num_updates_per_batch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--networks', type=int, default=1)
    parser.add_argument('--reward_scaling', type=float, default=5.0)
    parser.add_argument('--switch_cost_wrapper', type=int, default=1)
    parser.add_argument('--switch_cost', type=float, default=1.0)
    parser.add_argument('--max_time_between_switches', type=float, default=0.1)
    parser.add_argument('--time_as_part_of_state', type=int, default=1)
    parser.add_argument('--num_final_evals', type=int, default=10)

    args = parser.parse_args()
    main(args)
