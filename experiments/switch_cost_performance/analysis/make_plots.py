import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from typing import NamedTuple, Dict

LEGEND_FONT_SIZE = 22
TITLE_FONT_SIZE = 30
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

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


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

envs = ['Pendulum', 'Greenhouse']

for index, env_name in enumerate(envs):
    cur_data = data[data['env_name'] == env_name]
    cur_data_reward_without_switch_cost = cur_data.groupby('switch_cost')['results/total_reward'].agg(['mean', 'std'])
    cur_data_reward_without_switch_cost = cur_data_reward_without_switch_cost.reset_index()

    cur_data_reward_with_switch_cost= cur_data.groupby('switch_cost')['results/reward_with_switch_cost'].agg(['mean', 'std'])
    cur_data_reward_with_switch_cost = cur_data_reward_with_switch_cost.reset_index()

    baselines_reward: Dict[str, Statistics] = {}

    baselines_reward[
        r'Reward [Without Switch Cost]'] = Statistics(
        xs=np.array(cur_data_reward_without_switch_cost['switch_cost']),
        ys_mean=np.array(cur_data_reward_without_switch_cost['mean']),
        ys_std=np.array(cur_data_reward_without_switch_cost['std'])
    )

    baselines_reward[
        r'Reward [With Switch Cost]'] = Statistics(
        xs=np.array(cur_data_reward_with_switch_cost['switch_cost']),
        ys_mean=np.array(cur_data_reward_with_switch_cost['mean']),
        ys_std=np.array(cur_data_reward_with_switch_cost['std'])
    )


    for baseline_name, baseline_stat in baselines_reward.items():
        ax[index].plot(baseline_stat.xs, baseline_stat.ys_mean, label=baseline_name)
        ax[index].fill_between(baseline_stat.xs,
                           baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                           baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                           alpha=0.2)

    # ax[index].set_yscale('log')
    ax[index].set_title(f'{env_name}', fontsize=LABEL_FONT_SIZE)
    ax[index].set_xlabel(r'Switch Cost', fontsize=LABEL_FONT_SIZE)
    ax[index].set_ylabel('Reward', fontsize=LABEL_FONT_SIZE)


handles, labels = [], []
for axs in ax:
    for handle, label in zip(*axs.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=2,
           loc='upper center',
           # bbox_to_anchor=(0.5, 0.89),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)


plt.show()
