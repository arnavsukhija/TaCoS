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

NUM_SAMPLES_PER_SEED = 10

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

SWITCH_COST = 0.1  # [0.1, 1, 2, 3]
# MAX_TIME_BETWEEN_SWITCHES = 0.015
NUM_EVALS = 1
# MIN_TIME = 0.015 / 10

statistics = 'mean' # Can be median or mean


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray


def update_baselines(cur_data: pd.DataFrame,
                     baseline_name: str,
                     cur_baselines_reward_with_switch_cost: Dict[str, Statistics],
                     cur_baselines_reward_without_switch_cost: Dict[str, Statistics], ):
    # Identify columns that follow the pattern 'total_reward_{index}'
    columns_to_mean = [col for col in cur_data.columns if col.startswith('results/total_reward_')]

    # Compute the mean of these columns and add as a new column
    cur_data['results/total_reward'] = cur_data[columns_to_mean].mean(axis=1)

    # Identify columns that follow the pattern 'total_reward_{index}'
    columns_to_mean = [col for col in cur_data.columns if col.startswith('results/num_actions_')]

    # Compute the mean of these columns and add as a new column
    cur_data['results/num_actions'] = cur_data[columns_to_mean].mean(axis=1)

    grouped_data = cur_data.groupby('new_integration_dt')[f'results/total_reward'].agg([statistics, 'std'])
    grouped_data = grouped_data.reset_index()

    cur_baselines_reward_without_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data['new_integration_dt']),
        ys_mean=np.array(grouped_data[statistics]),
        ys_std=np.array(grouped_data['std'])
    )

    cur_data['results/reward_with_switch_cost'] = cur_data['results/total_reward'] - SWITCH_COST * cur_data[
        'results/num_actions']
    grouped_data_with_switch_cost = cur_data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
        [statistics, 'std'])
    grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

    cur_baselines_reward_with_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
        ys_mean=np.array(grouped_data_with_switch_cost[statistics]),
        ys_std=np.array(grouped_data_with_switch_cost['std'])
    )
    return cur_baselines_reward_with_switch_cost, cur_baselines_reward_without_switch_cost


baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data_adaptive = pd.read_csv('data/rccar/switch_cost.csv')
# data_adaptive = data_adaptive[data_adaptive['new_integration_dt'] >= MIN_TIME]
filtered_df = data_adaptive[data_adaptive['switch_cost'] == SWITCH_COST]



for index in range(NUM_EVALS):
    filtered_df[f'results/reward_with_switch_cost_{index}'] = filtered_df[
                                                                  f'results/total_reward_{index}'] - SWITCH_COST * \
                                                              filtered_df[f'results/num_actions_{index}']






baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=filtered_df,
    baseline_name="Switch-Cost-CTRL",
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

for baseline_name, baseline_stat in baselines_reward_without_switch_cost.items():
    ax[0].plot(baseline_stat.xs, baseline_stat.ys_mean, label=baseline_name)
    ax[0].fill_between(baseline_stat.xs,
                       baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                       baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                       alpha=0.2)

ax[0].set_xscale('log')
ax[0].set_xlabel(r'Integration dt', fontsize=LABEL_FONT_SIZE)
ax[0].set_ylabel('Reward [Without Switch Cost]', fontsize=LABEL_FONT_SIZE)

for baseline_name, baseline_stat in baselines_reward_with_switch_cost.items():
    ax[1].plot(baseline_stat.xs, baseline_stat.ys_mean, label=baseline_name)
    ax[1].fill_between(baseline_stat.xs,
                       baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                       baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                       alpha=0.2)

ax[1].set_xscale('log')
ax[1].set_xlabel(r'Integration dt', fontsize=LABEL_FONT_SIZE)
ax[1].set_ylabel('Reward [With Switch Cost]', fontsize=LABEL_FONT_SIZE)

handles, labels = [], []
for axs in ax:
    for handle, label in zip(*axs.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=2,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.89),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.suptitle(f'RCCar [Duration = 4 sec], [Switch Cost = {SWITCH_COST}]',
             fontsize=TITLE_FONT_SIZE,
             y=0.95)
fig.tight_layout(rect=[0.0, 0.0, 1, 0.8])
plt.savefig('rccar_switch_cost_varying_integration_dt.pdf')
plt.show()
