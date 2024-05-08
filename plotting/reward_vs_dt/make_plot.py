import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from typing import NamedTuple, Dict, Any

LEGEND_FONT_SIZE = 22
TITLE_FONT_SIZE = 30
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

NUM_SAMPLES_PER_SEED = 5
LINE_WIDTH = 5

BASELINE_NAMES = {
    'basline0': 'Switch-Cost-CTRL',
    'basline1': 'Same Compute \n Same Real Time \n More \# measurements',
    'basline2': 'More Compute \n Same Real Time \n More \# measurements',
    'basline3': 'Same Compute \n Less Real Time \n Same \# measurements',
}

LINESTYLES = {
    'basline0': 'solid',
    'basline1': 'dashed',
    'basline2': 'dotted',
    'basline3': 'dashdot',
}

COLORS = {
    'basline0': 'C0',
    'basline1': 'C1',
    'basline2': 'C2',
    'basline3': 'C3',
}

LINESTYLES_FROM_NAMES = {BASELINE_NAMES[name]: style for name, style in LINESTYLES.items()}
COLORS_FROM_NAMES = {BASELINE_NAMES[name]: color for name, color in COLORS.items()}

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

SWITCH_COST = 0.1  # [0.1, 1, 2, 3]
MAX_TIME_BETWEEN_SWITCHES = 0.5
NUM_EVALS = 10


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray


systems: Dict[str, Any] = {}


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

    grouped_data = cur_data.groupby('new_integration_dt')[f'results/total_reward'].agg(['mean', 'std'])
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


########################## Reacher ##########################
#############################################################

baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data_adaptive = pd.read_csv('data/reacher/switch_cost.csv')
filtered_df = data_adaptive[data_adaptive['switch_cost'] == SWITCH_COST]
for index in range(NUM_EVALS):
    filtered_df[f'results/reward_with_switch_cost_{index}'] = filtered_df[
                                                                  f'results/total_reward_{index}'] - SWITCH_COST * \
                                                              filtered_df[f'results/num_actions_{index}']

data_equidistant = pd.read_csv('data/reacher/no_switch_cost.csv')
for index in range(NUM_EVALS):
    data_equidistant[f'results/reward_with_switch_cost_{index}'] = data_equidistant[
                                                                       f'results/total_reward_{index}'] - SWITCH_COST * \
                                                                   data_equidistant[f'results/num_actions_{index}']

data_same_gd = data_equidistant[data_equidistant['same_amount_of_gradient_updates'] == True]
data_more_gd = data_equidistant[data_equidistant['same_amount_of_gradient_updates'] == False]

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=filtered_df,
    baseline_name=BASELINE_NAMES['basline0'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_same_gd,
    baseline_name=BASELINE_NAMES['basline1'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_more_gd,
    baseline_name=BASELINE_NAMES['basline2'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

systems['Reacher \n [Duration = 4 sec]'] = baselines_reward_without_switch_cost


########################## RCCar ############################
#############################################################

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


baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data_adaptive = pd.read_csv('data/rccar/switch_cost.csv')
filtered_df = data_adaptive[(data_adaptive['switch_cost'] == SWITCH_COST) &
                            (data_adaptive['max_time_between_switches'] == MAX_TIME_BETWEEN_SWITCHES)]
filtered_df['results/reward_with_switch_cost'] = filtered_df['results/total_reward'] - SWITCH_COST * filtered_df[
    'results/num_actions']

data_equidistant = pd.read_csv('data/rccar/no_switch_cost.csv')
data_equidistant['results/reward_with_switch_cost'] = data_equidistant['results/total_reward'] - SWITCH_COST * \
                                                      data_equidistant['results/num_actions']

data_same_gd = data_equidistant[data_equidistant['same_amount_of_gradient_updates'] == True]
data_more_gd = data_equidistant[data_equidistant['same_amount_of_gradient_updates'] == False]

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=filtered_df,
    baseline_name=BASELINE_NAMES['basline0'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_same_gd,
    baseline_name=BASELINE_NAMES['basline1'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_more_gd,
    baseline_name=BASELINE_NAMES['basline2'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

systems['RC Car \n [Duration = 4 sec]'] = baselines_reward_without_switch_cost

########################## Halfcheetah ######################
#############################################################

SWITCH_COST = 2
MAX_TIME_BETWEEN_SWITCHES = 0.05

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
########################################################################################
########################################################################################

grouped_data_adaptive = filtered_df.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data_adaptive = grouped_data_adaptive.reset_index()

baselines_reward_without_switch_cost[BASELINE_NAMES['basline0']] = Statistics(
    xs=np.array(grouped_data_adaptive['new_integration_dt']),
    ys_mean=np.array(grouped_data_adaptive['mean']),
    ys_std=np.array(grouped_data_adaptive['std'])
)

grouped_data_adaptive_with_switch_cost = filtered_df.groupby('new_integration_dt')[
    'results/reward_with_switch_cost'].agg(['mean', 'std'])
grouped_data_adaptive_with_switch_cost = grouped_data_adaptive_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[BASELINE_NAMES['basline0']] = Statistics(
    xs=np.array(grouped_data_adaptive_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_adaptive_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_adaptive_with_switch_cost['std'])
)

########################################################################################
########################################################################################

grouped_data = data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data = grouped_data.reset_index()

baselines_reward_without_switch_cost[
    BASELINE_NAMES['basline3']] = Statistics(
    xs=np.array(grouped_data['new_integration_dt']),
    ys_mean=np.array(grouped_data['mean']),
    ys_std=np.array(grouped_data['std'])
)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/total_steps']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[
    BASELINE_NAMES['basline3']] = Statistics(
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

baselines_reward_without_switch_cost[BASELINE_NAMES['basline2']] = Statistics(
    xs=np.array(grouped_data['new_integration_dt']),
    ys_mean=np.array(grouped_data['mean']),
    ys_std=np.array(grouped_data['std'])
)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/num_actions']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[BASELINE_NAMES['basline2']] = Statistics(
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

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data,
    baseline_name=BASELINE_NAMES['basline1'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost
)

systems['Halfcheetah \n [Duration = 10 sec]'] = baselines_reward_without_switch_cost

#############################################################
#############################################################

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

for index, (title, baselines) in enumerate(systems.items()):
    for baseline_name, baseline_stat in baselines.items():
        ax[index].plot(baseline_stat.xs, baseline_stat.ys_mean,
                       label=baseline_name,
                       linewidth=LINE_WIDTH,
                       linestyle=LINESTYLES_FROM_NAMES[baseline_name],
                       color=COLORS_FROM_NAMES[baseline_name],)
        ax[index].fill_between(baseline_stat.xs,
                               baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                               baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                               color=COLORS_FROM_NAMES[baseline_name],
                               alpha=0.2)

    ax[index].set_xscale('log')
    ax[index].set_xlabel(r'Integration dt', fontsize=LABEL_FONT_SIZE)
    ax[index].set_title(title, fontsize=TITLE_FONT_SIZE, pad=85)
    if index == 0:
        ax[index].set_ylabel('Reward [Without Switch Cost]', fontsize=LABEL_FONT_SIZE)

handles, labels = [], []
for axs in ax:
    for handle, label in zip(*axs.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=4,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.86),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.tight_layout()
plt.savefig('varying_integration_dt.pdf')
plt.show()
