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

SWITCH_COST = 2
MAX_TIME_BETWEEN_SWITCHES = 0.05


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray


baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data = pd.read_csv('data/halfcheetah/equidistant.csv')
data = data[data['new_integration_dt'] >= 0.05 / 30]
data_adaptive = pd.read_csv('data/halfcheetah/adaptive.csv')
filtered_df = data_adaptive[(data_adaptive['switch_cost'] == SWITCH_COST) &
                            (data_adaptive['max_time_between_switches'] == MAX_TIME_BETWEEN_SWITCHES) &
                            (data_adaptive['time_as_part_of_state'] == True)]
filtered_df['results/reward_with_switch_cost'] = filtered_df['results/total_reward'] - SWITCH_COST * filtered_df[
    'results/num_actions']


data_low_freq = pd.read_csv('data/halfcheetah/low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data_low_freq['results/total_reward'] = data_low_freq['results/total_reward_0']
filtered_df = pd.concat([filtered_df, data_low_freq])


########################################################################################
########################################################################################

grouped_data_adaptive = filtered_df.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data_adaptive = grouped_data_adaptive.reset_index()

baselines_reward_without_switch_cost[r'Switch-Cost-CTRL [Episodes=5000, GD updates=$10^6$, Measurements=$10^6$]'] = Statistics(
    xs=np.array(grouped_data_adaptive['new_integration_dt']),
    ys_mean=np.array(grouped_data_adaptive['mean']),
    ys_std=np.array(grouped_data_adaptive['std'])
)

grouped_data_adaptive_with_switch_cost = filtered_df.groupby('new_integration_dt')[
    'results/reward_with_switch_cost'].agg(['mean', 'std'])
grouped_data_adaptive_with_switch_cost = grouped_data_adaptive_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[r'Switch-Cost-CTRL [Episodes=5000, GD updates=$10^6$, Measurements=$10^6$]'] = Statistics(
    xs=np.array(grouped_data_adaptive_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_adaptive_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_adaptive_with_switch_cost['std'])
)

########################################################################################
########################################################################################

grouped_data = data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data = grouped_data.reset_index()

baselines_reward_without_switch_cost[r'Standard RL [Episodes=5000 $\times \frac{\text{Integration }dt}{0.05}$, GD updates=$10^6$, Measurements=$10^6$]'] = Statistics(
    xs=np.array(grouped_data['new_integration_dt']),
    ys_mean=np.array(grouped_data['mean']),
    ys_std=np.array(grouped_data['std'])
)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/total_steps']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[r'Standard RL [Episodes=5000 $\times \frac{\text{Integration }dt}{0.05}$, GD updates=$10^6$, Measurements=$10^6$]'] = Statistics(
    xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_with_switch_cost['std'])
)

######### Baseline: Same number of episodes, 1 grad update per env step #########
# data = pd.read_csv('data/halfcheetah/same_number_of_episodes.csv')
data = pd.read_csv('data/halfcheetah/no_switch_cost.csv')
data = data[data['same_amount_of_gradient_updates'] == False]

grouped_data = data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data = grouped_data.reset_index()

baselines_reward_without_switch_cost[r'Standard RL [Episodes=5000, GD updates=$10^6\times \frac{0.05}{\text{Integration }dt}$, Measurements=$10^6\times \frac{0.05}{\text{Integration }dt}$]'] = Statistics(
    xs=np.array(grouped_data['new_integration_dt']),
    ys_mean=np.array(grouped_data['mean']),
    ys_std=np.array(grouped_data['std'])
)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/num_actions']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[r'Standard RL [Episodes=5000, GD updates=$10^6\times \frac{0.05}{\text{Integration }dt}$, Measurements=$10^6\times \frac{0.05}{\text{Integration }dt}$]'] = Statistics(
    xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_with_switch_cost['std'])
)


def update_baselines(cur_data: pd.DataFrame,
                     baseline_name: str,
                     cur_baselines_reward_with_switch_cost: Dict[str, Statistics],
                     cur_baselines_reward_without_switch_cost: Dict[str, Statistics], ):
    grouped_data = cur_data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
    grouped_data = grouped_data.reset_index()

    cur_baselines_reward_without_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data['new_integration_dt']),
        ys_mean=np.array(grouped_data['mean']),
        ys_std=np.array(grouped_data['std'])
    )

    cur_data['results/reward_with_switch_cost'] = cur_data['results/total_reward'] - SWITCH_COST * cur_data[
        'results/num_actions']
    grouped_data_with_switch_cost = cur_data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
        ['mean', 'std'])
    grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

    cur_baselines_reward_with_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
        ys_mean=np.array(grouped_data_with_switch_cost['mean']),
        ys_std=np.array(grouped_data_with_switch_cost['std'])
    )
    return cur_baselines_reward_with_switch_cost, cur_baselines_reward_without_switch_cost


# data = pd.read_csv('data/halfcheetah/same_number_of_episodes_and_gradients.csv')
data = pd.read_csv('data/halfcheetah/no_switch_cost.csv')
data = data[data['same_amount_of_gradient_updates'] == True]

data_low_freq_pure_sac = pd.read_csv('data/halfcheetah/low_freq_pure_sac.csv')
data_low_freq_pure_sac['new_integration_dt'] = data_low_freq_pure_sac['new_integration_dt'] * data_low_freq_pure_sac['action_repeat']
data_low_freq_pure_sac['same_amount_of_gradient_updates'] = True
data_low_freq_pure_sac['results/total_reward'] = data_low_freq_pure_sac['results/total_reward_0']
data = pd.concat([data, data_low_freq_pure_sac])

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data,
    baseline_name=r'Standard RL [Episodes=5000, GD updates=$10^6$, Measurements=$10^6\times \frac{0.05}{\text{Integration }dt}$]',
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost
)

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
           ncols=1,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.89),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.suptitle(f'Halfcheetah run forward task [Duration = 10 sec], [Switch Cost = {SWITCH_COST}]',
             fontsize=TITLE_FONT_SIZE,
             y=0.95)
fig.tight_layout(rect=[0.0, 0.0, 1, 0.67])
plt.savefig('halfcheetah_switch_cost_varying_integration_dt.pdf')
plt.show()
