import argparse
import datetime
from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import wandb
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

from source.envs.linear_dynamics import LinearDynamicsBoundedSwitches
from source.utils.create_system_matrix import create_marginally_stable_matrix

PLOT_TRUE_TRAJECTORIES = True
ENTITY = 'trevenl'


def experiment(project_name: str,
               learning_discount_factor: int = 0.99,
               num_switches: int = 20,
               sac_train_steps: int = 100_000,
               wandb_logging: bool = True,
               plot_progress: bool = False,
               training_seed: int = 42,
               ):
    config = dict(learning_discount_factor=learning_discount_factor,
                  num_switches=num_switches,
                  sac_train_steps=sac_train_steps,
                  trainig_seed=training_seed)

    if wandb_logging:
        wandb.init(
            dir='/cluster/scratch/' + ENTITY,
            project=project_name,
            config=config,
        )

    key = jr.PRNGKey(42)
    x_dim, u_dim = 4, 4
    a = create_marginally_stable_matrix(x_dim, key=key)
    b = jnp.eye(u_dim)
    x0 = jr.uniform(shape=(x_dim,), key=key)

    exp_discount_factor = jnp.log(1 / learning_discount_factor) / 0.05
    env = LinearDynamicsBoundedSwitches(
        a=a,
        b=b,
        x0=x0,
        time_horizon=jnp.array(10.0),
        number_of_switches=jnp.array(num_switches),
        min_time_between_switches=jnp.array(0.1),
        max_time_between_switches=jnp.array(5.0),
        number_of_reward_evaluations=10,
        discount_factor=float(exp_discount_factor),
        reward_type='lqr'
    )

    optimizer = SAC(
        environment=env,
        num_timesteps=sac_train_steps,
        episode_length=num_switches,
        action_repeat=1,
        num_env_steps_between_updates=10,
        num_envs=4,
        num_eval_envs=32,
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
        grad_updates_per_step=10 * 32,
        deterministic_eval=True,
        init_log_alpha=0.,
        policy_hidden_layer_sizes=(64, 64),
        policy_activation=swish,
        critic_hidden_layer_sizes=(64, 64),
        critic_activation=swish,
        wandb_logging=wandb_logging,
        return_best_model=True,
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
    if plot_progress:
        policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(training_seed), progress_fn=progress)
    else:
        policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(training_seed))
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)

    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    # Evaluation; we make a new env without discounting for the evaluation

    env = LinearDynamicsBoundedSwitches(
        a=a,
        b=b,
        x0=x0,
        time_horizon=jnp.array(10.0),
        number_of_switches=jnp.array(20),
        min_time_between_switches=jnp.array(0.1),
        max_time_between_switches=jnp.array(5.0),
        number_of_reward_evaluations=10,
        discount_factor=0.0,
        reward_type='lqr',
    )

    if plot_progress:
        time_to_jit = times[1] - times[0]
        time_to_train = times[-1] - times[1]

        if wandb_logging:
            wandb.log({'time_to_jit': str(time_to_jit),
                       'time_to_train': str(time_to_train)})
        else:
            print(f'time to jit: {time_to_jit}')
            print(f'time to train: {time_to_train}')

    state = env.reset(key)
    trajectory = [state]
    actions = []
    for i in range(num_switches):
        u = policy(state.obs)[0]
        state = env.step(state, u)
        trajectory.append(state)
        actions.append(u)

    xs_augmented = jnp.stack([state.obs for state in trajectory])
    us = jnp.stack(actions)
    ts = xs_augmented[0, -2] - xs_augmented[:, -2]
    xs = xs_augmented[:, :-2]
    rewards = jnp.stack([state.reward for state in trajectory])

    LABEL_SIZE = 20
    TICKS_SIZE = 20

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=
    r'\usepackage{amsmath}'
    r'\usepackage{bm}'
    r'\def\vx{{\bm{x}}}'
    r'\def\vf{{\bm{f}}}')

    import matplotlib as mpl

    mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
    mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
    axs[0].plot(ts, xs)
    axs[0].set_xlabel('time', fontsize=LABEL_SIZE)
    axs[0].set_ylabel('State', fontsize=LABEL_SIZE)

    axs[1].step(ts, jnp.concatenate([us, us[-1].reshape(1, -1)]), where='post', label=r'$u$')
    axs[1].set_xlabel('Time', fontsize=LABEL_SIZE)
    axs[1].set_ylabel('Action', fontsize=LABEL_SIZE)

    integrated_rewards = rewards[1:] / jnp.diff(ts)
    axs[2].step(ts, jnp.concatenate([integrated_rewards, integrated_rewards[-1].reshape(1, )]),
                where='post', label='Rewards')
    axs[2].set_xlabel('Time', fontsize=LABEL_SIZE)
    axs[2].set_ylabel('Instance Reward', fontsize=LABEL_SIZE)

    axs[3].plot(jnp.diff(ts), label='Times for actions')
    axs[3].set_xlabel('Action Steps', fontsize=LABEL_SIZE)
    axs[3].set_ylabel('Time for action', fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.show()


def main(args):
    experiment(
        project_name=args.project_name,
        learning_discount_factor=args.learning_discount_factor,
        num_switches=args.num_switches,
        sac_train_steps=args.sac_train_steps,
        wandb_logging=bool(args.wandb_logging),
        plot_progress=bool(args.plot_progress),
        training_seed=args.training_seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='LinearWhenToControl')
    parser.add_argument('--learning_discount_factor', type=float, default=0.99)
    parser.add_argument('--num_switches', type=int, default=50)
    parser.add_argument('--sac_train_steps', type=int, default=40_000)
    parser.add_argument('--wandb_logging', type=int, default=1)
    parser.add_argument('--plot_progress', type=int, default=1)
    parser.add_argument('--training_seed', type=int, default=43)

    args = parser.parse_args()
    main(args)
