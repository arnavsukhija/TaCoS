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

from wtc.envs.greenhouse import GreenHouseEnv
from wtc.wrappers.bounded_switches import FixedNumOfSwitchesWrapper

PLOT_TRUE_TRAJECTORIES = True
ENTITY = 'trevenl'


def experiment(project_name: str,
               wrapper: bool = True,
               env_name: str = 'GreenHouse',
               num_switches: int = 10,
               sac_train_steps: int = 100_000,
               training_seed: int = 42,
               action_repeat: int = 1,
               plot_progress: bool = False,
               episode_length: int = 300,
               ) -> None:
    if env_name == 'GreenHouse':
        env = GreenHouseEnv()
        discount_factor = 0.99
        min_reps = 1
        max_reps = 50

    if wrapper:
        action_repeat = 1
        env = FixedNumOfSwitchesWrapper(env,
                                        num_integrator_steps=episode_length,
                                        num_switches=num_switches,
                                        discounting=discount_factor,
                                        min_time_between_switches=min_reps * env.dt,
                                        max_time_between_switches=max_reps * env.dt)

    assert episode_length % action_repeat == 0
    config = dict(wrapper=wrapper,
                  action_repeat=action_repeat,
                  episode_length=episode_length,
                  num_switches=num_switches,
                  min_reps=min_reps,
                  max_reps=max_reps,
                  sac_train_steps=sac_train_steps,
                  trainig_seed=training_seed,
                  plot_progress=plot_progress)

    wandb.init(
        dir='/cluster/scratch/' + ENTITY,
        project=project_name,
        config=config,
    )
    num_env_steps_between_updates = 10
    num_envs = 32

    optimizer = SAC(
        environment=env,
        num_timesteps=sac_train_steps,
        episode_length=episode_length,
        action_repeat=action_repeat,
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
        discounting=discount_factor,
        batch_size=32,
        num_evals=20,
        normalize_observations=True,
        reward_scaling=1.,
        tau=0.005,
        min_replay_size=10 ** 2,
        max_replay_size=sac_train_steps,
        grad_updates_per_step=num_env_steps_between_updates * num_envs,
        deterministic_eval=True,
        init_log_alpha=0.,
        policy_hidden_layer_sizes=(64, 64),
        policy_activation=swish,
        critic_hidden_layer_sizes=(64, 64),
        critic_activation=swish,
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
    if plot_progress:
        policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(training_seed), progress_fn=progress)
    else:
        policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(training_seed))
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)

    @jax.jit
    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    if env_name == 'GreenHouse':
        env = GreenHouseEnv()
        min_reps = 1
        max_reps = 50

    if wrapper:
        env = FixedNumOfSwitchesWrapper(env,
                                        num_integrator_steps=episode_length,
                                        num_switches=num_switches,
                                        discounting=1.0,
                                        min_time_between_switches=min_reps * env.dt,
                                        max_time_between_switches=max_reps * env.dt)

    if wrapper:
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
        all_states = []
        while not state.done:
            state, one_traj = step(state, None)
            one_traj, full_trajectory = one_traj[:-1], one_traj[-1]
            trajectory.append(one_traj)
            full_trajectories.append(full_trajectory)
            all_states.append(state)

        trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
        full_trajectory = jtu.tree_map(lambda *xs: jnp.concatenate(xs), *full_trajectories)

        xs_full_trajectory = jnp.concatenate([init_state.obs[:-2].reshape(1, -1), full_trajectory.obs, ])
        rewards_full_trajectory = jnp.concatenate([init_state.reward.reshape(1, ), full_trajectory.reward])
        ts_full_trajectory = jnp.arange(0, xs_full_trajectory.shape[0]) * env.env.dt

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))

        us = trajectory[1][:, :-1]
        times = trajectory[0][:, -2]

        # All times are the times when we ended the actions
        all_ts = init_state.obs[-2] - times
        all_ts = jnp.concatenate([jnp.array([0.0]), all_ts])

        for i in range(env.observation_size - 2):
            axs[0].plot(ts_full_trajectory, xs_full_trajectory[:, i])
        for h in all_ts[:-1]:
            axs[0].axvline(x=h, color='black', ls='--', alpha=0.4)

        axs[0].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[0].set_ylabel('State', fontsize=LABEL_SIZE)
        axs[0].set_ylim([-50, 50])

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
        plt.savefig('pendulum_switch_cost.pdf')
        plt.show()
        print(f'Total reward: {jnp.sum(trajectory[2])}')
        print(f'Total number of actions {len(us)}')

        for ax in axs:
            ax.legend(fontsize=LEGEND_SIZE)
        plt.tight_layout()

        wandb.log({'pendulum_switch_bound': wandb.Image(fig),
                   'results/total_reward': float(jnp.sum(trajectory[2])),
                   'results/num_actions': trajectory[0].shape[0]})

    else:
        state = env.reset(rng=jr.PRNGKey(0))
        step_fn = jax.jit(env.step)
        trajectory = []
        total_steps = 0
        while (not state.done) and (total_steps < episode_length):
            action = policy(state.obs)[0]
            for _ in range(action_repeat):
                state = step_fn(state, action)
                total_steps += 1
                trajectory.append(state)

        trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)

        wandb.log({'results/total_reward': jnp.sum(trajectory.reward),
                   'results/num_actions': len(trajectory.reward)})


def main(args):
    experiment(
        project_name=args.project_name,
        wrapper=bool(args.wrapper),
        env_name=args.env_name,
        num_switches=args.num_switches,
        sac_train_steps=args.sac_train_steps,
        training_seed=args.training_seed,
        action_repeat=args.action_repeat,
        plot_progress=bool(args.plot_progress),
        episode_length=args.episode_length,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='GreenhouseTest')
    parser.add_argument('--wrapper', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='GreenHouse')
    parser.add_argument('--num_switches', type=int, default=23)
    parser.add_argument('--sac_train_steps', type=int, default=40_000)
    parser.add_argument('--training_seed', type=int, default=43)
    parser.add_argument('--action_repeat', type=int, default=10)
    parser.add_argument('--plot_progress', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1)

    args = parser.parse_args()
    main(args)
