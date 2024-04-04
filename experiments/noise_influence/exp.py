import argparse
import datetime
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

from wtc.envs.pendulum import PendulumEnv
from wtc.envs.greenhouse import GreenHouseEnv
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import ConstantSwitchCost, IHSwitchCostWrapper

ENTITY = 'trevenl'


def experiment(
        env_name: str = 'Pendulum',
        project_name: str = 'TestPendulumNoiseInfluence',
        scale: float = 0.1,
        seed: int = 0,
        wrapper: bool = True,
        num_timesteps: int = 20_000,
) -> None:
    PLOT_TRUE_TRAJECTORIES = True
    episode_length = 200
    time_as_part_of_state = True

    if env_name == 'Pendulum':
        env = PendulumEnv(reward_source='dm-control',
                          add_process_noise=True,
                          process_noise_scale=scale * jnp.array([0.01, 0.01, 0.1]))

        min_time_between_switches = 1 * env.dt
        max_time_between_switches = 30 * env.dt

        discount_factor = 0.99
        continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                    dt=env.dt)
        if wrapper:
            env = IHSwitchCostWrapper(env,
                                      num_integrator_steps=episode_length,
                                      min_time_between_switches=1 * env.dt,
                                      max_time_between_switches=30 * env.dt,
                                      switch_cost=ConstantSwitchCost(value=jnp.array(0.1)),
                                      time_as_part_of_state=time_as_part_of_state,
                                      discounting=discount_factor)

    elif env_name == 'Greenhouse':
        env = GreenHouseEnv(add_process_noise=True,
                            process_noise_scale=jnp.array(scale))

        min_time_between_switches = 1 * env.dt
        max_time_between_switches = 30 * env.dt

        discount_factor = 0.997
        continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                    dt=env.dt)

        if wrapper:
            env = IHSwitchCostWrapper(env,
                                      num_integrator_steps=episode_length,
                                      min_time_between_switches=1 * env.dt,
                                      max_time_between_switches=30 * env.dt,
                                      switch_cost=ConstantSwitchCost(value=jnp.array(0.2)),
                                      time_as_part_of_state=time_as_part_of_state,
                                      discounting=discount_factor)

    config = dict(seed=seed,
                  scale=scale,
                  env_name=env_name,
                  wrapper=wrapper,
                  num_timesteps=num_timesteps
                  )

    wandb.init(
        project=project_name,
        dir='/cluster/scratch/' + ENTITY,
        config=config,
    )

    num_env_steps_between_updates = 5
    num_envs = 32
    if wrapper:
        optimizer = SAC(
            target_entropy=None,
            environment=env,
            num_timesteps=num_timesteps,
            episode_length=episode_length,
            action_repeat=1,
            num_env_steps_between_updates=num_env_steps_between_updates,
            num_envs=num_envs,
            num_eval_envs=32,
            lr_alpha=3e-4,
            lr_policy=3e-4,
            lr_q=3e-4,
            wd_alpha=0.,
            wd_policy=0.,
            wd_q=0.,
            max_grad_norm=1e5,
            discounting=discount_factor,
            batch_size=64,
            num_evals=20,
            normalize_observations=True,
            reward_scaling=1.,
            tau=0.005,
            min_replay_size=10 ** 3,
            max_replay_size=10 ** 5,
            grad_updates_per_step=num_env_steps_between_updates * num_envs,
            deterministic_eval=True,
            init_log_alpha=0.,
            policy_hidden_layer_sizes=(32,) * 5,
            policy_activation=swish,
            critic_hidden_layer_sizes=(128,) * 3,
            critic_activation=swish,
            wandb_logging=True,
            return_best_model=True,
            non_equidistant_time=True,
            continuous_discounting=continuous_discounting,
            min_time_between_switches=min_time_between_switches,
            max_time_between_switches=max_time_between_switches,
            env_dt=env.dt,
        )
    else:
        optimizer = SAC(
            target_entropy=None,
            environment=env,
            num_timesteps=num_timesteps,
            episode_length=episode_length,
            action_repeat=1,
            num_env_steps_between_updates=num_env_steps_between_updates,
            num_envs=num_envs,
            num_eval_envs=32,
            lr_alpha=3e-4,
            lr_policy=3e-4,
            lr_q=3e-4,
            wd_alpha=0.,
            wd_policy=0.,
            wd_q=0.,
            max_grad_norm=1e5,
            discounting=discount_factor,
            batch_size=64,
            num_evals=20,
            normalize_observations=True,
            reward_scaling=1.,
            tau=0.005,
            min_replay_size=10 ** 3,
            max_replay_size=10 ** 5,
            grad_updates_per_step=num_env_steps_between_updates * num_envs,
            deterministic_eval=True,
            init_log_alpha=0.,
            policy_hidden_layer_sizes=(32,) * 5,
            policy_activation=swish,
            critic_hidden_layer_sizes=(128,) * 3,
            critic_activation=swish,
            wandb_logging=True,
            return_best_model=True,
            non_equidistant_time=False,
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

    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    if wrapper:
        def step(state, _):
            u = policy(state.obs)[0]
            print('Step')
            print(f'Time to go {u[-1]}')
            next_state, rest = env.simulation_step(state, u)
            return next_state, (next_state.obs, u, next_state.reward, rest)

        if env_name == 'Pendulum':
            env = PendulumEnv(reward_source='dm-control',
                              add_process_noise=True,
                              process_noise_scale=scale * jnp.array([0.01, 0.01, 0.1]))
            env = IHSwitchCostWrapper(env,
                                      num_integrator_steps=episode_length,
                                      min_time_between_switches=1 * env.dt,
                                      max_time_between_switches=30 * env.dt,
                                      switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                      time_as_part_of_state=time_as_part_of_state,
                                      discounting=1.0)

        elif env_name == 'Greenhouse':
            env = GreenHouseEnv(add_process_noise=True,
                                process_noise_scale=jnp.array(scale))
            env = IHSwitchCostWrapper(env,
                                      num_integrator_steps=episode_length,
                                      min_time_between_switches=1 * env.dt,
                                      max_time_between_switches=30 * env.dt,
                                      switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                      time_as_part_of_state=time_as_part_of_state,
                                      discounting=1.0)

        state = env.reset(rng=jr.PRNGKey(0))
        init_state = state
        LEGEND_SIZE = 20
        LABEL_SIZE = 20
        TICKS_SIZE = 20

        import matplotlib as mpl

        mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
        mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

        trajectory = []
        full_trajectories = []
        all_states = []
        while not state.done:
            state, one_traj = step(state, None)
            one_traj, full_trajectory = one_traj[:-1], one_traj[-1]
            trajectory.append(one_traj)
            full_trajectories.append(full_trajectory)
            all_states.append(state)

        trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
        full_trajectory = jtu.tree_map(lambda *xs: jnp.concatenate(xs), *full_trajectories)
        all_states = jtu.tree_map(lambda *xs: jnp.stack(xs), *all_states)

        xs_full_trajectory = jnp.concatenate([init_state.obs[:-1].reshape(1, -1), full_trajectory.obs, ])
        rewards_full_trajectory = jnp.concatenate([init_state.reward.reshape(1, ), full_trajectory.reward])
        ts_full_trajectory = jnp.arange(0, xs_full_trajectory.shape[0]) * env.env.dt

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
        xs = trajectory[0][:, :-1]
        us = trajectory[1][:, :-1]
        rewards = trajectory[2]
        if time_as_part_of_state:
            times = trajectory[0][:, -1]
        else:
            times = all_states.pipeline_state.time

        # All times are the times when we ended the actions
        all_ts = times
        all_ts = jnp.concatenate([jnp.array([0.0]), all_ts])

        all_xs = jnp.concatenate([state.obs[:-1].reshape(1, -1), xs])

        state_dict = {0: r'cos($\theta$)',
                      1: r'sin($\theta$)',
                      2: r'$\omega$'}

        if PLOT_TRUE_TRAJECTORIES:
            for i in range(env.observation_size - 1):
                axs[0].plot(ts_full_trajectory, xs_full_trajectory[:, i])
            for h in all_ts[:-1]:
                axs[0].axvline(x=h, color='black', ls='--', alpha=0.4)
        else:
            for i in range(3):
                axs[0].plot(all_ts, all_xs[:, i], label=state_dict[i])

        axs[0].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[0].set_ylabel('State', fontsize=LABEL_SIZE)

        axs[1].step(all_ts, jnp.concatenate([us, us[-1].reshape(1, -1)]), where='post', label=r'$u$')
        axs[1].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[1].set_ylabel('Action', fontsize=LABEL_SIZE)

        integrated_rewards = rewards / jnp.diff(all_ts) * 0.05

        if PLOT_TRUE_TRAJECTORIES:
            axs[2].plot(ts_full_trajectory, rewards_full_trajectory, label='Rewards')
            for h in all_ts[:-1]:
                axs[2].axvline(x=h, color='black', ls='--', alpha=0.4)
        else:
            axs[2].step(all_ts, jnp.concatenate([integrated_rewards, integrated_rewards[-1].reshape(1, )]),
                        where='post', label='Rewards')
        axs[2].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[2].set_ylabel('Instance reward', fontsize=LABEL_SIZE)

        axs[3].plot(jnp.diff(all_ts), label='Times for actions')
        axs[3].set_xlabel('Action Steps', fontsize=LABEL_SIZE)
        axs[3].set_ylabel('Time for action', fontsize=LABEL_SIZE)

        for ax in axs:
            ax.legend(fontsize=LEGEND_SIZE)
        plt.tight_layout()

        print(f'Total reward: {jnp.sum(rewards_full_trajectory[:episode_length])}')
        print(f'Total number of actions {len(us)}')

        wandb.log({'pendulum_switch_bound': wandb.Image(fig),
                   'results/total_reward': float(jnp.sum(trajectory[2])),
                   'results/num_actions': trajectory[0].shape[0],
                   'results/action_times': jnp.diff(all_ts)})

    else:
        if env_name == 'Pendulum':
            env = PendulumEnv(reward_source='dm-control',
                              add_process_noise=True,
                              process_noise_scale=scale * jnp.array([0.01, 0.01, 0.1]))

        elif env_name == 'Greenhouse':
            env = GreenHouseEnv(add_process_noise=True,
                                process_noise_scale=jnp.array(scale))

        state = env.reset(rng=jr.PRNGKey(0))
        step_fn = jax.jit(env.step)

        action_repeat = 1
        trajectory = []
        total_steps = 0
        while (not state.done) and (total_steps < episode_length):
            action = policy(state.obs)[0]
            for _ in range(action_repeat):
                state = step_fn(state, action)
                total_steps += 1
                trajectory.append(state)

        trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)

        wandb.log({'results/total_reward': float(jnp.sum(trajectory.reward)),
                   'results/num_actions': total_steps})

    wandb.finish()


def main(args):
    experiment(env_name=args.env_name,
               project_name=args.project_name,
               scale=args.scale,
               seed=args.seed,
               wrapper=bool(args.wrapper),
               num_timesteps=args.num_timesteps,
               )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pendulum')
    parser.add_argument('--project_name', type=str, default='TestPendulumNoiseInfluence')
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wrapper', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=20_000)

    args = parser.parse_args()
    main(args)
