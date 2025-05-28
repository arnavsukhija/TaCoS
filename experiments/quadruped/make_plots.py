import wandb
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
import pandas as pd


def fetch_run_data(project_name, max_time_repeat, switch_cost, penalize):
    """
    Fetches logged rewards and number of actions from Wandb runs with a specific tag.

    Args:
        project_name (str): The name of your Wandb project.
        max_time_repeat: The upper bound which interests us
        switch_cost: The switch cost we pay for action switches
        penalize: Bool which defines whether we penalize the interactions on the reward

    Returns:
        tuple: A tuple containing two lists:
               - all_rewards (list of float): A list of total rewards from each matching run.
               - all_num_actions (list of int): A list of total actions from each matching run.
               Returns empty lists if no matching runs are found or if the data is not logged.
    """
    api = wandb.Api()
    runs = api.runs(project_name)

    all_rewards = []
    all_num_actions = []

    for run in runs:
        summary = run.summary
        config = run.config
        if config['switch_cost'] != switch_cost or config['max_time_repeat'] != max_time_repeat or config[
            'num_timesteps'] != 400_000_000:
            continue
        all_rewards.append(summary["Results/Total reward"] - int(penalize) * switch_cost * summary[
            "Results/Number of actions"])  #we penalize using the interaction cost 0.1 from training
        all_num_actions.append(summary["Results/Number of actions"])

    return all_rewards, all_num_actions


def analyze_run_data(all_rewards, all_num_actions):
    """
    Computes the average and standard deviation of rewards and the average and standard deviation of the number of actions.

    Args:
        all_rewards (list of float): A list of total rewards.
        all_num_actions (list of int): A list of total number of actions.

    Returns:
        tuple: A tuple containing:
               - avg_reward (float or None): Average total reward, None if no rewards.
               - std_reward (float or None): Standard deviation of total reward, None if fewer than 2 rewards.
               - avg_num_actions (float or None): Average total number of actions, None if no action counts.
               - std_num_actions (float or None): Standard deviation of total number of actions, None if fewer than 2 action counts.
    """
    avg_reward = np.mean(all_rewards) if all_rewards else None
    stderr_reward = sem(all_rewards) if len(all_rewards) >= 2 else None
    avg_num_actions = np.mean(all_num_actions) if all_num_actions else None
    stderr_num_actions = sem(all_num_actions) if len(all_num_actions) >= 2 else None
    return avg_reward, stderr_reward, avg_num_actions, stderr_num_actions


def get_wandb_run_stats(project_name, max_time_repeat, switch_cost, penalize):
    """
    Fetches runs from Wandb based on a tag and computes the average and standard
    deviation of logged total rewards and the average and standard deviation of total actions.

    Args:
        project_name (str): The name of your Wandb project.
        max_time_repeat (int): the upper bound we are interested in.
        switch_cost (int): the switch cost we pay for action switches

    Returns:
        tuple: A tuple containing:
               - avg_reward (float or None): Average total reward.
               - std_reward (float or None): Standard deviation of total reward.
               - avg_num_actions (float or None): Average total number of actions.
               - std_num_actions (float or None): Standard deviation of total number of actions.
    """
    all_rewards, all_num_actions = fetch_run_data(project_name, max_time_repeat, switch_cost, penalize)
    avg_reward, std_reward, avg_num_actions, std_num_actions = analyze_run_data(all_rewards, all_num_actions)

    print(f"Analysis for runs with max time '{max_time_repeat}' and switch cost '{switch_cost}':")
    if avg_reward is not None:
        print(f"  Average Total Reward: {avg_reward:.2f}")
    else:
        print("  No reward data found")

    if std_reward is not None:
        print(f"  Standard Deviation of Total Reward: {std_reward:.2f}")
    else:
        print("  Standard deviation of reward cannot be computed (less than 2 data points).")

    if avg_num_actions is not None:
        print(f"  Average Total Number of Actions: {avg_num_actions:.2f}")
    else:
        print("  No action count data found")

    if std_num_actions is not None:
        print(f"  Standard Deviation of Total Number of Actions: {std_num_actions:.2f}")
    else:
        print("  Standard deviation of action count cannot be computed (less than 2 data points).")

    return avg_reward, std_reward, avg_num_actions, std_num_actions


PENALIZE = False
switch_cost = 0.005
mean_rewards = []
std_rewards = []
mean_actions = []
std_actions = []

## PPO (default max time is 5)
df = pd.read_csv('wandb_export_2025-05-16T11_39_47.666+02_00.csv')
ppo_reward_suffix = 'Results/Total reward '
ppo_reward_column_names = [col for col in df if col.endswith(ppo_reward_suffix)]

rewards = df[ppo_reward_column_names]
avg_reward = rewards.mean(axis=1).iloc[0]
std_reward = rewards.sem(axis=1).iloc[0]
avg_actions = 1000
std_action = 0.0

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(np.array(avg_actions))
std_actions.append(np.array(std_action))
## Tacos3
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats(
    'arnavsukhija-eth-zurich/TaCoSGo1JoystickFlatTerrain_1.0TacosDiscount', 3, switch_cost, PENALIZE)

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(avg_actions)
std_actions.append(std_action)

## Tacos4
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats(
    'arnavsukhija-eth-zurich/TaCoSGo1JoystickFlatTerrain_1.0TacosDiscount', 4, switch_cost, PENALIZE)

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(avg_actions)
std_actions.append(std_action)

## Tacos5
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats(
    'arnavsukhija-eth-zurich/TaCoSGo1JoystickFlatTerrain_1.0TacosDiscount', 5, switch_cost, PENALIZE)

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(avg_actions)
std_actions.append(std_action)

## Tacos10
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats(
    'arnavsukhija-eth-zurich/TaCoSGo1JoystickFlatTerrain_1.0TacosDiscount', 10, switch_cost, PENALIZE)

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(avg_actions)
std_actions.append(std_action)

algorithm_names = ['PPO', 'PPO-Tacos3', 'PPO-Tacos4', 'PPO-Tacos5', 'PPO-Tacos10']

data_to_plot = [
    {"data": mean_rewards, "title": "Reward on Evaluation Task", "xlabel": "Mean Reward"},
    {"data": mean_actions, "title": "Number of Actions on Evaluation Task", "xlabel": "Mean Number of Interactions"},
]

x_positions = np.arange(len(algorithm_names))  # Create numerical positions for the bars
bar_width = 0.7  # Adjust the width of the bars as needed

# --- Bar Plot for Mean Rewards with Standard Deviation Error Bars ---
plt.figure(figsize=(8, 6))
plt.bar(x_positions, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7, color='skyblue')
plt.xlabel("Agent")
plt.ylabel("Total Reward")
plt.title("Total Reward on Go1 evaluation task")
plt.xticks(x_positions, algorithm_names)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Bar Plot for Mean Actions with Standard Deviation Error Bars ---
plt.figure(figsize=(8, 6))
plt.bar(x_positions, mean_actions, yerr=std_actions, capsize=5, alpha=0.7, color='lightcoral')
plt.xlabel("Agent")
plt.ylabel("Number of Actions")
plt.title("Number of Actions on Go1 evaluation task")
plt.xticks(x_positions, algorithm_names)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
