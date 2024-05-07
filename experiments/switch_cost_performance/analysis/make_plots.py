import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from typing import NamedTuple, Dict

LEGEND_FONT_SIZE = 26
TITLE_FONT_SIZE = 34
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300
LINE_WIDTH = 4

NUM_SAMPLES_PER_SEED = 5

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

data = pd.read_csv('data/switch_cost_performance.csv')
data['results/reward_with_switch_cost'] = data['results/total_reward'] - data['switch_cost'] * data[
    'results/num_actions']


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray
    color: str = "Blue"
    linestyle: str = "--"
    linewidth: float = LINE_WIDTH


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

envs_labels = ['Pendulum Swing-up \n [Duration=10s]', 'Greenhouse Temperature Tracking \n [Duration=25h]', ]
envs = ['Pendulum', 'Greenhouse', ]

for index, env_name in enumerate(envs):
    cur_data = data[data['env_name'] == env_name]
    cur_data_reward_without_switch_cost = cur_data.groupby('switch_cost')['results/total_reward'].agg(['mean', 'std'])
    cur_data_reward_without_switch_cost = cur_data_reward_without_switch_cost.reset_index()

    cur_data_reward_with_switch_cost = cur_data.groupby('switch_cost')['results/reward_with_switch_cost'].agg(
        ['mean', 'std'])
    cur_data_reward_with_switch_cost = cur_data_reward_with_switch_cost.reset_index()

    cur_data_num_actions = cur_data.groupby('switch_cost')['results/num_actions'].agg(['mean', 'std'])
    cur_data_num_actions = cur_data_num_actions.reset_index()

    baselines_reward: Dict[str, Statistics] = {}
    baseline_num_actions: Dict[str, Statistics] = {}

    baseline_num_actions['\# Actions'] = Statistics(
        xs=np.array(cur_data_num_actions['switch_cost']),
        ys_mean=np.array(cur_data_num_actions['mean']),
        ys_std=np.array(cur_data_num_actions['std']),
        color='Green',
        linestyle='-'
    )

    baselines_reward[
        r'Reward [Without Switch Cost]'] = Statistics(
        xs=np.array(cur_data_reward_without_switch_cost['switch_cost']),
        ys_mean=np.array(cur_data_reward_without_switch_cost['mean']),
        ys_std=np.array(cur_data_reward_without_switch_cost['std']),
        color='C0',
        linestyle='-'
    )

    baselines_reward[
        r'Reward [With Switch Cost]'] = Statistics(
        xs=np.array(cur_data_reward_with_switch_cost['switch_cost']),
        ys_mean=np.array(cur_data_reward_with_switch_cost['mean']),
        ys_std=np.array(cur_data_reward_with_switch_cost['std']),
        color='C0',
        linestyle='dashed'
    )

    for baseline_name, baseline_stat in baselines_reward.items():
        ax[index].plot(baseline_stat.xs,
                       baseline_stat.ys_mean,
                       label=baseline_name,
                       color=baseline_stat.color,
                       linestyle=baseline_stat.linestyle,
                       linewidth=baseline_stat.linewidth)
        ax[index].fill_between(baseline_stat.xs,
                               baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                               baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                               alpha=0.2,
                               color=baseline_stat.color,
                               )
        ax[index].tick_params(axis='y', labelcolor=baseline_stat.color)
        ax[index].set_ylabel('Reward', fontsize=LABEL_FONT_SIZE, color=baseline_stat.color)

    ax[index].set_title(envs_labels[index], fontsize=LABEL_FONT_SIZE, pad=60)
    ax[index].set_xlabel(r'Switch Cost', fontsize=LABEL_FONT_SIZE)

    ax_right_side = ax[index].twinx()
    for baseline_name, baseline_stat in baseline_num_actions.items():
        ax_right_side.plot(baseline_stat.xs, baseline_stat.ys_mean,
                           label=baseline_name,
                           color=baseline_stat.color,
                           linestyle=baseline_stat.linestyle,
                           linewidth=baseline_stat.linewidth)
        ax_right_side.fill_between(baseline_stat.xs,
                                   baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                                   baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                                   alpha=0.2,
                                   color=baseline_stat.color,
                                   )
        ax_right_side.tick_params(axis='y', labelcolor=baseline_stat.color)
        ax_right_side.set_ylabel('\# Actions', fontsize=LABEL_FONT_SIZE, color=baseline_stat.color)

handles, labels = [], []
for axs in ax:
    for handle, label in zip(*axs.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
for handle, label in zip(*ax_right_side.get_legend_handles_labels()):
    handles.append(handle)
    labels.append(label)
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=3,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.85),
           fontsize=LEGEND_FONT_SIZE,
           frameon=True)

# fig.suptitle(f'Switch cost influence',
#              fontsize=TITLE_FONT_SIZE,
#              y=0.95)
fig.tight_layout(rect=[0.0, 0.0, 1, 1])
plt.savefig('switch_cost_influence.pdf')
plt.show()
