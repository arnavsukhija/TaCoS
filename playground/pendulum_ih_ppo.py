import datetime
from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from jax.lax import scan
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.ppo.ppo_brax_env import PPO

from wtc.envs.pendulum import PendulumEnv
from wtc.envs.pendulum_swing_down import PendulumEnv as PendulumEnvSwingDown
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import ConstantSwitchCost, IHSwitchCostWrapper

if __name__ == "__main__":
    wrapper = False
    PLOT_TRUE_TRAJECTORIES = True
    swing_up = True
    action_repeat = 1
    episode_length = 100
    time_as_part_of_state = False

    if swing_up:
        env = PendulumEnv(reward_source='dm-control')
    else:
        env = PendulumEnvSwingDown(reward_source='dm-control')

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
                                  switch_cost=ConstantSwitchCost(value=jnp.array(1.0)),
                                  time_as_part_of_state=time_as_part_of_state)

    else:
        action_repeat = 1

    num_envs = 2048
    unroll_length = 20
    batch_size = 1024
    entropy_cost = 1e0
    num_minibatches = 2 * num_envs // batch_size
    num_updates_per_batch = 4

    optimizer = PPO(
        environment=env,
        num_timesteps=1_000_000,
        episode_length=episode_length,
        action_repeat=action_repeat,
        num_envs=num_envs,
        num_eval_envs=4,
        lr=3e-4,
        wd=0.,
        entropy_cost=entropy_cost,
        unroll_length=unroll_length,
        # max_grad_norm=1e5, TODO: implement this
        discounting=discount_factor,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        num_evals=20,
        normalize_observations=True,
        reward_scaling=10.,
        clipping_epsilon=0.3,
        gae_lambda=0.95,
        deterministic_eval=True,
        policy_hidden_layer_sizes=(64,) * 3,
        policy_activation=swish,
        critic_hidden_layer_sizes=(64,) * 3,
        critic_activation=swish,
        wandb_logging=False,
        normalize_advantage=True,
        # return_best_model=True, TODO: implement this
        non_equidistant_time=True,
        continuous_discounting=continuous_discounting,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches,
        env_dt=env.dt,
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
    # wandb.init(
    #     project='TestIHSwitchCost'
    # )
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(0), progress_fn=progress)
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)


    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))


    # @jax.jit
    def step(state, _):
        u = policy(state.obs)[0]
        print('Step')
        print(f'Time to go {u[-1]}')
        next_state, rest = env.simulation_step(state, u)
        return next_state, (next_state.obs, u, next_state.reward, rest)


    state = env.reset(rng=jr.PRNGKey(0))

    if wrapper:
        init_state = state
        LEGEND_SIZE = 20
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

        xs_full_trajectory = jnp.concatenate([init_state.obs[:3].reshape(1, -1), full_trajectory.obs, ])
        rewards_full_trajectory = jnp.concatenate([init_state.reward.reshape(1, ), full_trajectory.reward])
        # ts_full_trajectory = jnp.linspace(0, env.time_horizon, episode_length)
        ts_full_trajectory = jnp.arange(0, xs_full_trajectory.shape[0]) * env.env.dt

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
        xs = trajectory[0][:, :-1]
        us = trajectory[1][:, :-1]
        rewards = trajectory[2]
        if time_as_part_of_state:
            times = trajectory[0][:, -1]
        else:
            times = all_states.pipeline_state.time
        times_for_actions = trajectory[1][:, -1]

        total_time = env.time_horizon
        # All times are the times when we ended the actions
        all_ts = times
        all_ts = jnp.concatenate([jnp.array([0.0]), all_ts])

        all_xs = jnp.concatenate([state.obs[:-1].reshape(1, -1), xs])

        state_dict = {0: r'cos($\theta$)',
                      1: r'sin($\theta$)',
                      2: r'$\omega$'}

        if PLOT_TRUE_TRAJECTORIES:
            for i in range(3):
                axs[0].plot(ts_full_trajectory, xs_full_trajectory[:, i], label=state_dict[i])
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

        # axs[4].plot(times_to_go, label='Time to go')
        # axs[3].plot(times_for_actions, label='Time for actions NORMALIZED')
        for ax in axs:
            ax.legend(fontsize=LEGEND_SIZE)
        plt.tight_layout()
        plt.savefig('pendulum_switch_cost.pdf')
        plt.show()
        print(f'Total reward: {jnp.sum(rewards_full_trajectory[:episode_length])}')
        print(f'Total number of actions {len(us)}')

    else:
        horizon = episode_length

        num_steps = horizon // action_repeat


        def repeated_step(state, _):
            u = policy(state.obs)[0]

            def f(state, _):
                nstate = env.step(state, u)
                return nstate, nstate.reward

            state, rewards = scan(f, state, (), action_repeat)
            state = state.replace(reward=jnp.sum(rewards, axis=0))
            return state, (state.obs, u, state.reward)


        x_last, trajectory = scan(repeated_step, state, None, length=num_steps)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
        axs[0].plot(trajectory[0], label='Xs')
        axs[0].legend()
        axs[1].plot(trajectory[1], drawstyle='steps-post', label='Us')
        axs[1].legend()
        axs[2].plot(trajectory[2], label='Rewards')
        axs[2].legend()
        plt.show()
        print(f'Total reward: {jnp.sum(trajectory[2])}')

    time_to_jit = times[1] - times[0]
    time_to_train = times[-1] - times[1]

    print(f'time to jit: {time_to_jit}')
    print(f'time to train: {time_to_train}')
