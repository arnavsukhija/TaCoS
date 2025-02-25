import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, List
import os

transition_cost = 1
NUM_SAMPLES = 5
EPISODE_TIME = 4.0
STEPS_PER_UPDATE = 50_000 / 20
BASE_DT_DIVISOR = 80
BASE_DT = 0.5

env = 'rccar'

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

    all_lengths = np.mean(all_lengths, axis=0)
    if add_transition_cost:
        xs = STEPS_PER_UPDATE / all_lengths * EPISODE_TIME
    else:
        xs = STEPS_PER_UPDATE * BASE_DT_DIVISOR / all_lengths * EPISODE_TIME
    xs = np.cumsum(xs)
    xs = xs - xs[0]

    stats = Statistics(
        xs=xs,
        ys_mean=np.mean(all_rewards, axis=0),
        ys_std=np.std(all_rewards, axis=0),
        name=baseline_name
    )
    statistics.append(stats)
    return statistics


stats = []

baselines = {
    'baseline0': 'SAC',
    'baseline1': 'SAC_MC',
    'baseline2': 'SAC_TAC',
}

folder_path = 'data/rccar/'

data_path = f'data/{env}/high_freq_sac_tac.csv'
data = pd.read_csv(data_path)
data = data[data['switch_cost'] == 1]
stats = update_statistics(
    data=data,
    baseline_name=baselines['baseline2'],
    statistics=stats,
    add_transition_cost=True
)

data_path = f'data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == True]
stats = update_statistics(
    data=data,
    baseline_name=baselines['baseline0'],
    statistics=stats,
    add_transition_cost=False
)

data_path = f'data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == False]
stats = update_statistics(
    data=data,
    baseline_name=baselines['baseline1'],
    statistics=stats,
    add_transition_cost=False
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
ax.set_xlabel('Physical Time')
ax.set_ylabel('Episode reward')
plt.legend()
plt.show()
