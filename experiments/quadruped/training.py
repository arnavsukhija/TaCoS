import argparse
import datetime
import os
from typing import Tuple

import cloudpickle
import pickle
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb

from jax.nn import swish

from wtc.agents.ppo.ppo_brax_env import PPO
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import ConstantSwitchCost, IHSwitchCostWrapper

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params


from jax import config

config.update("jax_debug_nans", True)

ENTITY = 'asukhija'

def save_policy(policy_params):
    if wandb.run is None:
        raise RuntimeError("wandb.run is not initialized. Ensure wandb.init() is called before logging artifacts.")

    # Ensure the 'Policies' directory inside the wandb run directory exists
    directory = os.path.join(os.getcwd(), 'Policies')
    if not os.path.exists(directory):
        os.makedirs(directory)

    policy_path = os.path.join(directory, f"policy_params_{wandb.run.id}.pkl")

    try:
        # 1️⃣ Inspect policy_params
        print("Inspecting policy_params:", policy_params)

        # 2️⃣ Save policy to local storage
        with open(policy_path, "wb") as f:
            cloudpickle.dump(policy_params, f)

        # 3️⃣ Check file size
        file_size = os.path.getsize(policy_path)
        print(f"File size: {file_size} bytes")

        # 4️⃣ Attempt to load the file back to verify it
        try:
            with open(policy_path, "rb") as f:
                loaded_policy = cloudpickle.load(f)
            print("Successfully loaded policy from file for verification.")
        except Exception as e:
            print(f"Error loading policy file for verification: {e}")
            return  # Stop the upload if the file is invalid.

        # 5️⃣ Ensure file exists before uploading to wandb
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"File not found: {policy_path}")

        wandb.save(policy_path, wandb.run.dir)

        print(f"Successfully saved and uploaded {policy_path} to Weights & Biases.")

    except Exception as e:
        print(f"An error occurred during policy upload: {e}")
    print("Policy saved to wandb!")

def save_trajectory(full_trajectory, index):
    if wandb.run is None:
        raise RuntimeError("wandb.run is not initialized. Ensure wandb.init() is called before logging artifacts.")

    # Ensure the 'Trajectories' directory inside the wandb run directory exists
    directory = os.path.join(os.getcwd(), 'Trajectories')
    os.makedirs(directory, exist_ok=True)

    trajectory_path = os.path.join(directory, f"trajectory_{index}_{wandb.run.id}.pkl")

    try:
        # 1️⃣ Inspect trajectory data
        print(f"Inspecting trajectory {index}:", full_trajectory)

        # 2️⃣ Save trajectory to local storage
        with open(trajectory_path, "wb") as f:
            cloudpickle.dump(full_trajectory, f)

        # 3️⃣ Check file size
        file_size = os.path.getsize(trajectory_path)
        print(f"File size: {file_size} bytes")

        # 4️⃣ Attempt to load the file back to verify it
        try:
            with open(trajectory_path, "rb") as f:
                loaded_trajectory = cloudpickle.load(f)
            print(f"Successfully loaded trajectory {index} from file for verification.")
        except Exception as e:
            print(f"Error loading trajectory file {index} for verification: {e}")
            return  # Stop the upload if the file is invalid.

        # 5️⃣ Ensure file exists before uploading to wandb
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"File not found: {trajectory_path}")

        # 6️⃣ Ensure WandB tracks the file
        wandb.save(trajectory_path, trajectory_path)  # Explicitly track the file before logging

        print(f"Successfully saved and uploaded trajectory {index} to Weights & Biases.")

    except Exception as e:
        print(f"An error occurred during trajectory upload: {e}")

    print(f"Trajectory {index} saved to wandb!")
def experiment(env_name: str = 'Go1JoystickFlatTerrain',
               backend: str = 'generalized',
               project_name: str = 'GPUSpeedTest',
               seed: int = 0,
               num_eval_envs: int = 128,
               switch_cost_wrapper: bool = False,
               switch_cost: float = 0.1,
               max_time_repeat: int = 10,
               min_time_repeat: int = 1,
               time_as_part_of_state: bool = True,
               num_final_evals: int = 10,
               ):
    # we load the env from the playground and read out the params
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_config = dict(ppo_params)
    action_repeat = ppo_config['action_repeat']
    batch_size = ppo_config['batch_size']
    discounting = ppo_config['discounting']
    entropy_cost = ppo_config['entropy_cost']
    episode_length = ppo_config['episode_length']
    learning_rate = ppo_config['learning_rate']
    max_grad_norm = ppo_config['max_grad_norm']
    policy_hidden_layer_sizes = ppo_config['network_factory']['policy_hidden_layer_sizes']
    critic_hidden_layer_sizes = ppo_config['network_factory']['value_hidden_layer_sizes']
    value_obs_key = ppo_config['network_factory']['value_obs_key']
    policy_obs_key = ppo_config['network_factory']['policy_obs_key']
    normalize_observations = ppo_config['normalize_observations']
    num_envs = ppo_config['num_envs']
    num_evals = ppo_config['num_evals']
    num_minibatches = ppo_config['num_minibatches']
    num_resets_per_eval = ppo_config['num_resets_per_eval']
    num_timesteps = ppo_config['num_timesteps']
    num_updates_per_batch = ppo_config['num_updates_per_batch']
    reward_scaling = ppo_config['reward_scaling']
    unroll_length = ppo_config['unroll_length']
    sim_dt = env_cfg['sim_dt']
    ctrl_dt = env_cfg['ctrl_dt']

    if switch_cost_wrapper:
        continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discounting,
                                                                    dt=ctrl_dt)

        env = IHSwitchCostWrapper(env=env,
                                  num_integrator_steps=episode_steps,
                                  min_time_between_switches=min_time_repeat * env_dt,
                                  # Hardcoded to be at least the integration step
                                  max_time_between_switches=max_time_repeat * env_dt,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(switch_cost)),
                                  discounting=discounting,
                                  time_as_part_of_state=time_as_part_of_state,
                                  ismujoco_env=True,
                                  )


    config = dict(env_name=env_name,
                  backend=backend,
                  num_timesteps=num_timesteps,
                  episode_time=episode_time,
                  new_integration_dt=env.dt,
                  new_episode_steps=episode_time // env.dt,
                  base_discount_factor=base_discount_factor,
                  new_discount_factor=new_discount_factor,
                  seed=seed,
                  num_envs=num_envs,
                  num_eval_envs=num_eval_envs,
                  entropy_cost=entropy_cost,
                  unroll_length=unroll_length,
                  num_minibatches=num_minibatches,
                  num_updates_per_batch=num_updates_per_batch,
                  policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                  critic_hidden_layer_sizes=critic_hidden_layer_sizes,
                  batch_size=batch_size,
                  reward_scaling=reward_scaling,
                  switch_cost_wrapper=switch_cost_wrapper,
                  switch_cost=switch_cost,
                  max_time_repeat=max_time_repeat,
                  time_as_part_of_state=time_as_part_of_state,
                  num_final_evals=num_final_evals,
                  min_time_repeat=min_time_repeat,
                  )
    if switch_cost_wrapper:
        wandb.init(
            project=project_name,
            group=f"max_actions{max_time_repeat}",
            dir='/cluster/scratch/' + ENTITY,
            config=config,
        )
    else:
        wandb.init(
            project=project_name,
            dir='/cluster/scratch/' + ENTITY,
            config=config,
        )
    if switch_cost_wrapper: #using the interaction cost TaCoS in this case, since we have wrapped the environment using the switch cost wrapper (augmented state, reward, steps)
        optimizer = PPO(
            environment=env, #passing switch cost env
            num_timesteps=num_timesteps,
            episode_length=episode_steps,
            action_repeat=action_repeat, #number of times we repeat action before evaluation
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=lr,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=new_discount_factor,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            num_evals=num_evals,
            normalize_observations=normalize_observations,
            reward_scaling=reward_scaling,
            max_grad_norm=max_grad_norm,
            clipping_epsilon=0.3, #clipping for PPO objective
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
            min_time_between_switches=min_time_repeat * env_dt, #can be set to 1/30
            max_time_between_switches=max_time_repeat * env_dt, #can be set to 1
            env_dt=env.dt,  #best is 1/30
        )
    else: #standard PPO with discount factor adaptation for continuous tasks, improves performance on continuous tasks.
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
            discounting=base_discount_factor,
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

    save_policy(policy_params)
    print("Policy saved to wandb!")
    @jax.jit
    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    ########################## Evaluation ##########################
    ################################################################

    print(f'Starting with evaluation')
    if switch_cost_wrapper:
        if env_name == 'rccar':
            # Episode time needs to be 4.0 seconds
            env = RCCar(margin_factor=20, sample_init_pos=False, domain_randomization=False) # No domain randomization while evaluation and no initial pos sampling

        if action_delay > 0.0:
            env = ActionDelayWrapper(env, action_delay)
        env = IHSwitchCostWrapper(env=env,
                                  num_integrator_steps=episode_steps,
                                  min_time_between_switches=min_time_repeat * env.dt,
                                  max_time_between_switches=max_time_repeat * env.dt,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  discounting=new_discount_factor,
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
            save_trajectory(full_trajectory, index)
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

        print("End of plotting, saving figure locally to path")
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))

        filename = "switch_bound_figure.png"
        fig.savefig(filename)
        full_path = os.path.abspath(filename)
        print(f"File saved locally at {full_path}")
        print('End of plotting, uploading results to wandb')

        wandb.log({'switch_bound_figure': wandb.Image(fig), })

        print('Results uploaded to wandb')

    else:
        if env_name == 'rccar':
            env = RCCar(margin_factor=20, sample_init_pos=False)

        if action_delay > 0.0:
            env = ActionDelayWrapper(env, action_delay)

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
               episode_steps=args.episode_steps,
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
               max_time_repeat=args.max_time_repeat,
               time_as_part_of_state=bool(args.time_as_part_of_state),
               num_final_evals=args.num_final_evals,
               min_time_repeat=args.min_time_repeat,
               domain_randomization=args.domain_randomization,
               sample_init_pos=args.sample_init_pos,
               action_delay = args.action_delay
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='rccar')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='GPUSpeedTest')
    parser.add_argument('--num_timesteps', type=int, default=100_000)
    parser.add_argument('--episode_steps', type=int, default=200)
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
    parser.add_argument('--max_time_repeat', type=int, default=5)
    parser.add_argument('--min_time_repeat', type=int, default=1)
    parser.add_argument('--time_as_part_of_state', type=int, default=1)
    parser.add_argument('--num_final_evals', type=int, default=10)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--num_env_steps_between_updates', type=int, default=10)
    parser.add_argument('--same_amount_of_gradient_updates', type=int, default=1,
                        help='Flag for consistent gradient updates.')
    parser.add_argument('--domain_randomization', type=int, default=1)
    parser.add_argument('--sample_init_pos', type=int, default=1)
    parser.add_argument('--action_delay', type=float, default=0.0)

    args = parser.parse_args()
    main(args)
