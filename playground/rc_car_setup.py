import datetime
from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax.lax import scan
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

from wtc.envs.rccar import RCCar

if __name__ == "__main__":
    action_repeat = 1
    base_dt = 0.5
    base_episode_length = 7
    discounting = 0.9
    dt_divisor = 1

    dt = base_dt / dt_divisor
    episode_length = base_episode_length * dt_divisor

    env = RCCar(margin_factor=20, dt=dt)
    num_env_steps_between_updates = 10
    num_envs = 32
    optimizer = SAC(
        environment=env,
        num_timesteps=500_000,
        episode_length=episode_length,
        action_repeat=action_repeat,
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
        discounting=0.99,
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
        policy_hidden_layer_sizes=(64, 64),
        policy_activation=swish,
        critic_hidden_layer_sizes=(64, 64),
        critic_activation=swish,
        wandb_logging=False,
        return_best_model=True,
        non_equidistant_time=False,
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
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(0), progress_fn=progress)
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)


    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))


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


    state = env.reset(rng=jr.PRNGKey(0))
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
