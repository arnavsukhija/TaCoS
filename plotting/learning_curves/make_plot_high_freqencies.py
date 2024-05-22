import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, List
import os

systems_eval = dict()

######################### RC Car #########################
###########################################################################

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

systems_eval['RC Car'] = stats

######################### Reacher #########################
###########################################################################

transition_cost = 0.1
NUM_SAMPLES = 5
EPISODE_TIME = 2.0
STEPS_PER_UPDATE = 100_000 / 20
BASE_DT_DIVISOR = 10


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

data_path = 'data/reacher/high_freq_sac_tac.csv'
data = pd.read_csv(data_path)
stats = update_statistics(
    data=data,
    baseline_name=baselines['baseline2'],
    statistics=stats,
    add_transition_cost=True
)

data_path = 'data/reacher/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == True]
stats = update_statistics(
    data=data,
    baseline_name=baselines['baseline0'],
    statistics=stats,
    add_transition_cost=False
)

data_path = 'data/reacher/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == False]
stats = update_statistics(
    data=data,
    baseline_name=baselines['baseline1'],
    statistics=stats,
    add_transition_cost=False
)

systems_eval['Reacher'] = stats

######################### Halfcheetah #########################
###########################################################################

transition_cost = 2
NUM_SAMPLES = 5
EPISODE_TIME = 10.0
STEPS_PER_UPDATE = 1_000_000 / 20
BASE_DT_DIVISOR = 10
BASE_DT = 0.05

env = 'halfcheetah'


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
data = data[data['switch_cost'] == transition_cost]
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

systems_eval['Halfcheetah'] = stats

######################### Humanoid #########################
###########################################################################

transition_cost = 0.1
NUM_SAMPLES = 5
EPISODE_TIME = 3.0
STEPS_PER_UPDATE = 5_000_000 / 20
BASE_DT_DIVISOR = 5
BASE_DT = 0.015

env = 'humanoid'


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
        xs = STEPS_PER_UPDATE * BASE_DT * np.arange(1, len(all_lengths) + 1)
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

systems_eval['Humanoid'] = stats

###########################################################################
###########################################################################
LEGEND_FONT_SIZE = 28
TITLE_FONT_SIZE = 33
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

NUM_SAMPLES_PER_SEED = 5
LINE_WIDTH = 5

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vu{{\bm{u}}}'
r'\def\vf{{\bm{f}}}')
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
handles, labels = [], []

for index, (system, stats) in enumerate(systems_eval.items()):
    for stat in stats:
        axs[index].plot(stat.xs, stat.ys_mean, label=stat.name,
                        linewidth=LINE_WIDTH, )
        axs[index].fill_between(
            stat.xs,
            stat.ys_mean - stat.ys_std / np.sqrt(NUM_SAMPLES),
            stat.ys_mean + stat.ys_std / np.sqrt(NUM_SAMPLES),
            alpha=0.2,
        )
    if system == 'Humanoid':
        axs[index].set_xlim(0, 100_000)
    axs[index].set_xlabel('Physical Time [sec]', fontsize=LABEL_FONT_SIZE)
    if index == 0:
        axs[index].set_ylabel('Episode reward', fontsize=LABEL_FONT_SIZE)
    axs[index].set_title(system, fontsize=TITLE_FONT_SIZE)


    for handle, label in zip(*axs[index].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=5,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.05),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.tight_layout(rect=[0.0, 0.0, 1, 0.9])
plt.show()
