import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from typing import NamedTuple, Tuple

LEGEND_FONT_SIZE = 26
TITLE_FONT_SIZE = 26
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

EPISODE_LEN = 300
NUMBER_OF_SAMPLES = 5
LINE_WIDTH = 5

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray
    title: str
    ax_lim: Tuple[int, int]


stats_optimized = []
stats_equidistant = []

############################# Greenhouse tracking #############################
###############################################################################

data = pd.read_csv('data/greenhouse_temperature_tracking.csv')
data_bounded_switches = data.loc[data['wrapper'] == True]
data_repeated_actions = data.loc[data['wrapper'] == False]

data_repeated_actions.loc[:, 'actual_switches'] = EPISODE_LEN / data_repeated_actions['action_repeat']

# Prepare data for data_bounded_switches
grouped_bounded_switches = data_bounded_switches.groupby('num_switches')['results/total_reward'].agg(['mean', 'std'])
grouped_bounded_switches = grouped_bounded_switches.reset_index()

xs_bounded_switches = np.array(grouped_bounded_switches['num_switches'])
ys_bounded_switches_mean = np.array(grouped_bounded_switches['mean'])
ys_bounded_switches_std = np.array(grouped_bounded_switches['std'])

# Prepare data for data_repeated_actions
grouped_repeated_actions = data_repeated_actions.groupby('actual_switches')['results/total_reward'].agg(
    ['mean', 'std'])
grouped_repeated_actions = grouped_repeated_actions.reset_index()

xs_repeated_actions = np.array(grouped_repeated_actions['actual_switches'])
ys_repeated_actions_mean = np.array(grouped_repeated_actions['mean'])
ys_repeated_actions_std = np.array(grouped_repeated_actions['std'])

greenhouse_stats_optimized = Statistics(
    xs=xs_bounded_switches,
    ys_mean=ys_bounded_switches_mean,
    ys_std=ys_bounded_switches_std,
    title='Greenhouse Temperature Tracking \n [Duration=25h]',
    ax_lim=(0, 35)
)
greenhouse_stats_equidistant = Statistics(
    xs=xs_repeated_actions,
    ys_mean=ys_repeated_actions_mean,
    ys_std=ys_repeated_actions_std,
    title='Greenhouse Temperature Tracking \n [Duration=25h]',
    ax_lim=(0, 35)
)

stats_optimized.append(greenhouse_stats_optimized)
stats_equidistant.append(greenhouse_stats_equidistant)

############################# Pendulum Swing Up #############################
###############################################################################

data = pd.read_csv('data/pendulum_swing_up.csv')
data_bounded_switches = data.loc[data['wrapper'] == True]
data_repeated_actions = data.loc[data['wrapper'] == False]

# Prepare data for data_bounded_switches
grouped_bounded_switches = data_bounded_switches.groupby('num_switches')['results/total_reward'].agg(['mean', 'std'])
grouped_bounded_switches = grouped_bounded_switches.reset_index()

xs_bounded_switches = np.array(grouped_bounded_switches['num_switches'])
ys_bounded_switches_mean = np.array(grouped_bounded_switches['mean'])
ys_bounded_switches_std = np.array(grouped_bounded_switches['std'])

# Prepare data for data_repeated_actions
grouped_repeated_actions = data_repeated_actions.groupby('results/num_actions')['results/total_reward'].agg(
    ['mean', 'std'])
grouped_repeated_actions = grouped_repeated_actions.reset_index()

xs_repeated_actions = np.array(grouped_repeated_actions['results/num_actions'])
ys_repeated_actions_mean = np.array(grouped_repeated_actions['mean'])
ys_repeated_actions_std = np.array(grouped_repeated_actions['std'])

stat_optimized = Statistics(
    xs=xs_bounded_switches,
    ys_mean=ys_bounded_switches_mean,
    ys_std=ys_bounded_switches_std,
    title='Pendulum Swing-up \n [Duration=10seconds]',
    ax_lim=(8, 35)
)
stat_equidistant = Statistics(
    xs=xs_repeated_actions,
    ys_mean=ys_repeated_actions_mean,
    ys_std=ys_repeated_actions_std,
    title='Pendulum Swing-up \n [Duration=10seconds]',
    ax_lim=(8, 35)
)

stats_optimized.append(stat_optimized)
stats_equidistant.append(stat_equidistant)

############################# Pendulum Swing Down #############################
###############################################################################

data = pd.read_csv('data/pendulum_swing_down.csv')
data_bounded_switches = data.loc[data['wrapper'] == True]
data_repeated_actions = data.loc[data['wrapper'] == False]

# Prepare data for data_bounded_switches
grouped_bounded_switches = data_bounded_switches.groupby('num_switches')['results/total_reward'].agg(['mean', 'std'])
grouped_bounded_switches = grouped_bounded_switches.reset_index()

xs_bounded_switches = np.array(grouped_bounded_switches['num_switches'])
ys_bounded_switches_mean = np.array(grouped_bounded_switches['mean'])
ys_bounded_switches_std = np.array(grouped_bounded_switches['std'])

# Prepare data for data_repeated_actions
grouped_repeated_actions = data_repeated_actions.groupby('results/num_actions')['results/total_reward'].agg(
    ['mean', 'std'])
grouped_repeated_actions = grouped_repeated_actions.reset_index()

xs_repeated_actions = np.array(grouped_repeated_actions['results/num_actions'])
ys_repeated_actions_mean = np.array(grouped_repeated_actions['mean'])
ys_repeated_actions_std = np.array(grouped_repeated_actions['std'])


stat_optimized = Statistics(
    xs=xs_bounded_switches,
    ys_mean=ys_bounded_switches_mean,
    ys_std=ys_bounded_switches_std,
    title='Pendulum Swing-down \n [Duration=15seconds]',
    ax_lim=(3, 35)
)
stat_equidistant = Statistics(
    xs=xs_repeated_actions,
    ys_mean=ys_repeated_actions_mean,
    ys_std=ys_repeated_actions_std,
    title='Pendulum Swing-down \n [Duration=15seconds]',
    ax_lim=(3, 35)
)

stats_optimized.append(stat_optimized)
stats_equidistant.append(stat_equidistant)

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=len(stats_optimized), figsize=(20, 5))
axs = np.array(axs).reshape(len(stats_optimized), )

for index, (stat_opt, stat_equi) in enumerate(zip(stats_optimized, stats_equidistant)):
    axs[index].plot(stat_opt.xs,
                    stat_opt.ys_mean,
                    label='Optimized time between interactions',
                    linewidth=LINE_WIDTH)
    axs[index].fill_between(stat_opt.xs,
                            stat_opt.ys_mean - 2 * stat_opt.ys_std / np.sqrt(NUMBER_OF_SAMPLES),
                            stat_opt.ys_mean + 2 * stat_opt.ys_std / np.sqrt(NUMBER_OF_SAMPLES),
                            alpha=0.3)

    axs[index].plot(stat_equi.xs, stat_equi.ys_mean,
                    label='Equidistant time between interactions',
                    linewidth=LINE_WIDTH,
                    linestyle='dashed'
                    )
    axs[index].fill_between(stat_equi.xs,
                            stat_equi.ys_mean - 2 * stat_equi.ys_std / np.sqrt(NUMBER_OF_SAMPLES),
                            stat_equi.ys_mean + 2 * stat_equi.ys_std / np.sqrt(NUMBER_OF_SAMPLES),
                            alpha=0.3)

    axs[index].set_xlim(*stat_equi.ax_lim)
    axs[index].set_xlabel('\# Interactions', fontsize=LABEL_FONT_SIZE)
    if index == 0:
        axs[index].set_ylabel('Episode reward', fontsize=LABEL_FONT_SIZE)
    axs[index].set_title(stat_opt.title, fontsize=TITLE_FONT_SIZE, pad=60)


handles, labels = [], []
for ax in axs:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=3,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.85),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.tight_layout(rect=[0.0, 0.0, 1, 1])

plt.savefig('reward_vs_number_of_actions.pdf')
plt.show()
