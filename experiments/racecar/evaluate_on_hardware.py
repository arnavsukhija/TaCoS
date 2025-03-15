import os
import time
from typing import NamedTuple, Dict, Any

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jax.nn import swish

from mbpo.optimizers.policy_optimizers.ppo.ppo_brax_env import PPO
from wtc.utils.discounting import discrete_to_continuous_discounting

from wtc.envs.hardware_setup.car_env_hardware import CarEnv
from wtc.envs.rccar import plot_rc_trajectory, RCCar

from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost

from wtc.wrappers.ih_switching_cost_gym import IHSwitchCostWrapper as IHSwitchCostGym, ConstantSwitchCost as ConstantSwitchCostGym

ENTITY = 'asukhija'


class RunSpec(NamedTuple):
    group_name: str
    run_id: str
    reward_config: dict | None = None


def run_all_policies_from_wandb(project_name: str, entity: str):
    """
    Retrieves all run configurations and policies from a WandB project,
    and runs them using run_with_learned_policy.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}")

    for run in runs:
        print(f"Running policy from run: {run.id}")
        config = run.config
        policy_artifact = run.logged_artifacts("policy", type="model")

        if not policy_artifact:
            print(f"Skipping run {run.id}: No policy artifact found.")
            continue

        policy_artifact = policy_artifact[0]
        policy_dir = policy_artifact.download()
        policy_path = os.path.join(policy_dir, "Policies", "tacos_policy.pkl")

        # Load the policy parameters from the pickle file

        ## We also save the observation, actions and step counts of each run
        observations = []
        actions = []
        step_counts = []
        try:
            with open(policy_path, 'rb') as f:
                policy_params = pickle.load(f)
        except FileNotFoundError:
            print(f"Skipping run {run.id}: Policy file not found at {policy_path}")
            continue
        except pickle.UnpicklingError:
            print(f"Skipping run {run.id}: Error unpickling policy file at {policy_path}")
            continue
        ## run the current policy with the run configuration
        try:
            observation, action, steps = run_with_learned_policy(
                policy_params=policy_params,
                project_name=project_name,
                group_name=run.group,
                run_id=run.id,
                config = config
            )
            observations.append(observation)
            actions.append(action)
            step_counts.append(steps)
            print(f"Successfully ran policy from run: {run.id}")
        except Exception as e:
            print(f"Error running policy from run {run.id}: {e}")

    observations = np.array(observations)
    actions = np.array(actions)
    step_counts = np.array(step_counts)
    return observations, actions, step_counts


def run_with_learned_policy(policy_params,
                            project_name: str,
                            group_name: str,
                            run_id: str,
                            config: Dict[str, Any],
                            ):
    """
    Num stacked frames: 3
    """
    """Fixed parameters from training"""
    encode_angle = True
    control_time_ms = 28.5

    """Configuration of the current run"""
    time_as_part_of_state = config.get("time_as_part_of_state", True)
    switch_cost_wrapper = config.get("switch_cost_wrapper", True)
    discount_factor = config.get("new_discount_factor",0.9)
    training_dt = config.get("new_integration_dt", 1/30)
    min_time_between_switches = config.get("min_time_repeat", 1) * training_dt
    max_time_between_switches = config.get("max_time_repeat", 5) * training_dt
    switch_cost = config.get("switch_cost", 0.1)
    episode_steps = config.get('new_episode_steps', 200)
    batch_size = config.get('batch_size', 1024)
    entropy_cost = config.get('entropy_cost', 0.01)
    episode_time = config.get('episode_time', 200 * training_dt)
    networks = config.get('networks', 0)
    num_envs = config.get('num_envs', 1024)
    num_timesteps = config.get('num_timesteps', 2_000_000)
    seed = config.get('seed', 0)
    reward_scaling = config.get('reward_scaling', 1.0)
    num_eval_envs = config.get('num_eval_envs', 32)
    unroll_length = config.get('unroll_length', 10)
    num_minibatches = config.get('num_minibatches', 32)
    num_updates_per_batch = config.get('num_updates_per_batch', 4)

    action_dim = 2 + int(switch_cost_wrapper) # includes time component now
    state_dim = 6 + int(encode_angle) + int(time_as_part_of_state)

    wandb.init(
        project=project_name,
        group=group_name,
        entity=ENTITY,
        id=run_id + 'f',
        resume="allow",
    )

    # determine actor and critic architecture size
    if networks == 0:
        policy_hidden_layer_sizes = (32,) * 5
        critic_hidden_layer_sizes = (128,) * 4
    elif networks == 1:
        policy_hidden_layer_sizes = (256,) * 2
        critic_hidden_layer_sizes = (256,) * 4
    else:
        policy_hidden_layer_sizes = (64, 64)
        critic_hidden_layer_sizes = (64, 64)

    ## this was the environment used for training, we use it to prepare the policy
    env = RCCar(margin_factor=20)
    ## we load the policy first
    if switch_cost_wrapper:
        # wrap using wtc SwitchCostWrapper
        env = IHSwitchCostWrapper(env=env,
                                  num_integrator_steps=episode_steps,
                                  min_time_between_switches=min_time_between_switches,
                                  # Hardcoded to be at least the integration step
                                  max_time_between_switches=max_time_between_switches,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(switch_cost)),
                                  discounting=discount_factor,
                                  time_as_part_of_state=time_as_part_of_state,
                                  )

        continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                    dt=training_dt) # used while training

        optimizer = PPO(
            environment=env,  # passing switch cost env
            num_timesteps= num_timesteps, #using the same as training
            episode_length=episode_steps,
            action_repeat=1,  # number of times we repeat action before evaluation
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=3e-4,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=discount_factor,
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
            min_time_between_switches=min_time_between_switches,
            max_time_between_switches=max_time_between_switches,
            env_dt=training_dt,
        )
    else:
        optimizer = PPO(
            environment=env,
            num_timesteps=num_timesteps,
            episode_length=episode_steps,
            action_repeat=1,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=3e-4,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=discount_factor,
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

    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)

    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))[0]

    ## we now prepare the simulation
    env = CarEnv(car_id=2, encode_angle=encode_angle, max_throttle=0.4, control_time_ms=control_time_ms,
                 num_frame_stacks=0)

    if switch_cost_wrapper:
        env = IHSwitchCostGym(env=env,
                                  min_time_between_switches=min_time_between_switches,
                                  max_time_between_switches=max_time_between_switches,
                                  switch_cost=ConstantSwitchCostGym(value=0.0),
                                  # since we are using an evaluation mode, no switch cost
                                  discounting=discount_factor,  # should this be the same?
                                  time_as_part_of_state=time_as_part_of_state)  # this was done while training

    obs, _ = env.reset()
    observations = []
    actions = []
    rewards = []
    step_count = 0

    """
    Simulate the car on the learned policy
    """
    for i in range(200):
        action = np.array(policy(obs))
        actions.append(action)
        obs, reward, terminate, info = env.step(action)
        step_count += 1
        observations.append(obs)
        rewards.append(reward)

        if terminate:
            break


    print('We end with simulation')
    env.close()
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)

    reward_from_observations = np.sum(rewards)
    reward_terminal = info['terminal_reward']
    total_reward = reward_from_observations + reward_terminal
    print('Terminal reward: ', reward_terminal)
    print('Reward from observations: ', reward_from_observations)
    print('Total reward: ', total_reward)

    wandb.log({
        "terminal_reward": reward_terminal,
        "reward_from_observations": reward_from_observations,
        "total_reward": total_reward,
        "number of actions": step_count
    })
    # We plot the trajectory
    fig, axes = plot_rc_trajectory(observations[:, :state_dim - int(time_as_part_of_state)], # remove time component from states
                                   actions[:, :action_dim - int(switch_cost_wrapper)], #remove time component from the actions
                                   encode_angle=encode_angle,
                                   show=True)
    wandb.log({'Trajectory on real rc car': wandb.Image(fig)})
    wandb.finish()
    return observations, actions, step_count

if __name__ == '__main__':
    import pickle

    obs, acts, steps = run_all_policies_from_wandb('project_name', ENTITY)
    print(obs)
    print(acts)
    print(steps)


