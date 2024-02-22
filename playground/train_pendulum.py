import datetime
from datetime import datetime

import jax.random as jr
import matplotlib.pyplot as plt
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC
from jax.lax import scan
import jax.numpy as jnp

from source.envs.pendulum import PendulumEnv
from source.wrappers.fixed_num_of_switches import FixedNumOfSwitchesWrapper

if __name__ == "__main__":
    wrapper = True
    env = PendulumEnv(reward_source='dm-control')
    action_repeat = 1

    if wrapper:
        num_switches = 40
        env = FixedNumOfSwitchesWrapper(env,
                                        num_integrator_steps=200,
                                        num_switches=num_switches,
                                        min_time_between_switches=1 * env.dt,
                                        max_time_between_switches=20 * env.dt)

    else:
        action_repeat = 10

    optimizer = SAC(
        environment=env,
        num_timesteps=100_000,
        episode_length=200,
        action_repeat=action_repeat,
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
        discounting=0.99,
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
        wandb_logging=False,
        return_best_model=True,
    )
    # train_fn = functools.partial(ppo.train,
    #                              num_timesteps=10_000_000,
    #                              num_evals=20,
    #                              reward_scaling=10,
    #                              episode_length=200,
    #                              normalize_observations=True,
    #                              action_repeat=1,
    #                              unroll_length=10,
    #                              num_minibatches=32,
    #                              num_updates_per_batch=4,
    #                              discounting=0.97,
    #                              learning_rate=3e-4,
    #                              entropy_cost=1e-2,
    #                              num_envs=2048,
    #                              batch_size=1024,
    #                              network_factory=functools.partial(ppo_networks.make_ppo_networks,
    #                                                                policy_hidden_layer_sizes=(32,) * 4,
    #                                                                value_hidden_layer_sizes=(256,) * 5,
    #                                                                activation=jax.nn.swish),
    #                              seed=1)

    # train_fn = functools.partial(sac.train,
    #                              num_timesteps=100_000,
    #                              num_evals=20,
    #                              reward_scaling=5,
    #                              episode_length=200,
    #                              normalize_observations=True,
    #                              action_repeat=1,
    #                              discounting=0.99,
    #                              learning_rate=3e-4,
    #                              num_envs=64,
    #                              batch_size=128,
    #                              grad_updates_per_step=64,
    #                              max_devices_per_host=1,
    #                              max_replay_size=2 ** 14,
    #                              min_replay_size=2 ** 7,
    #                              network_factory=functools.partial(sac_networks.make_sac_networks,
    #                                                                hidden_layer_sizes=(256, 256),
    #                                                                activation=jax.nn.swish),
    #                              seed=1)

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
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(0), progress_fn=progress)
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)


    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))


    def step(state, _):
        u = policy(state.obs)[0]
        next_state = env.step(state, u)
        return next_state, (next_state.obs, u, next_state.reward)


    state = env.reset(rng=jr.PRNGKey(0))

    if wrapper:
        horizon = num_switches
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

        x_last, trajectory = scan(step, state, None, length=horizon)
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
        xs = trajectory[0][:, :-2]
        us = trajectory[1][:, :-1]
        rewards = trajectory[2]
        times_to_go = trajectory[0][:, -2]
        times_for_actions = trajectory[1][:, -1]

        total_time = env.time_horizon
        # All times are the times when we ended the actions
        all_ts = total_time - times_to_go
        all_ts = jnp.concatenate([jnp.array([0.0]), all_ts])

        all_xs = jnp.concatenate([state.obs[:-2].reshape(1, -1), xs])

        state_dict = {0: r'cos($\theta$)',
                      1: r'sin($\theta$)',
                      2: r'$\omega$'}

        for i in range(3):
            axs[0].plot(all_ts, all_xs[:, i], label=state_dict[i])

        axs[0].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[0].set_ylabel('State', fontsize=LABEL_SIZE)

        axs[1].step(all_ts, jnp.concatenate([us, us[-1].reshape(1, -1)]), where='post', label=r'$u$')
        axs[1].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[1].set_ylabel('Action', fontsize=LABEL_SIZE)

        integrated_rewards = rewards / jnp.diff(all_ts) * 0.05

        axs[2].step(all_ts, jnp.concatenate([integrated_rewards, integrated_rewards[-1].reshape(1, )]),
                    where='post', label='Rewards')
        axs[2].set_xlabel('Time', fontsize=LABEL_SIZE)
        axs[2].set_ylabel('Instance reward', fontsize=LABEL_SIZE)

        axs[3].plot(jnp.diff(all_ts)[:-1], label='Times for actions')
        axs[3].set_xlabel('Action Steps', fontsize=LABEL_SIZE)
        axs[3].set_ylabel('Time for action', fontsize=LABEL_SIZE)

        # axs[4].plot(times_to_go, label='Time to go')
        # axs[3].plot(times_for_actions, label='Time for actions NORMALIZED')
        for ax in axs:
            ax.legend(fontsize=LEGEND_SIZE)
        plt.tight_layout()
        plt.savefig('pendulum.pdf')
        plt.show()
        print(f'Total reward: {jnp.sum(trajectory[2])}')

    else:
        horizon = 200

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
