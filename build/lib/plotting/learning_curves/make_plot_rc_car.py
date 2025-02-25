import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, List
import os

transition_cost = 0.1
NUM_SAMPLES = 5


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray
    name: str


def update_statistics(
        data: pd.DataFrame,
        baseline_name: str,
        statistics: List[Statistics],
        add_transition_cost: bool
):
    lengths_and_rewards = data['lengths_and_rewards']

    all_rewards = []
    all_lengths = []

    for tuple in lengths_and_rewards:
        tuple = eval(tuple)
        lengths, rewards = tuple
        all_rewards.append(np.array(rewards))
        all_lengths.append(np.array(lengths))

    all_rewards = np.stack(all_rewards)
    all_lengths = np.stack(all_lengths)

    if add_transition_cost:
        all_rewards = all_rewards + all_lengths * transition_cost

    stats = Statistics(
        xs=np.arange(all_rewards.shape[1]),
        ys_mean=np.mean(all_rewards, axis=0),
        ys_std=np.std(all_rewards, axis=0),
        name=baseline_name
    )
    statistics.append(stats)
    return statistics


stats = []

baselines = {
    'low_freq_sac.csv': ('SAC', True),
    'low_freq_sac_tac.csv': ('SAC-TAC', True),
    'low_freq_ppo_tac.csv': ('PPO-TAC', True),
}

folder_path = 'data/rccar/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    data = pd.read_csv(file_path)
    baseline_name, add_transition_cost = baselines[filename]
    stats = update_statistics(
        data=data,
        baseline_name=baseline_name,
        statistics=stats,
        add_transition_cost=add_transition_cost
    )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

for stat in stats:
    ax.plot(stat.xs, stat.ys_mean, label=stat.name)
    ax.fill_between(
        stat.xs,
        stat.ys_mean - stat.ys_std / np.sqrt(NUM_SAMPLES),
        stat.ys_mean + stat.ys_std / np.sqrt(NUM_SAMPLES),
        alpha=0.2,
    )
plt.legend()
plt.show()
