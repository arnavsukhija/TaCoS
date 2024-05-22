from typing import NamedTuple, Dict, Any
import matplotlib.gridspec as gridspec

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEGEND_FONT_SIZE = 28
TITLE_FONT_SIZE = 33
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

NUM_SAMPLES_PER_SEED = 5
LINE_WIDTH = 5

BASELINE_NAMES = {
    'basline0': r'\textsc{SAC-TaCoS}',
    'basline1': r'\textsc{SAC}',
    'basline2': r'\textsc{SAC-MC}',
    'basline3': 'SAC-LESS_PHYSICAL_TIME',
    'basline4': r'\textsc{PPO-TaCoS}',
}

LINESTYLES = {
    'basline0': 'solid',
    'basline1': 'dashed',
    'basline2': (0, (3, 1, 1, 1)),
    'basline3': 'dashdot',
    'basline4': 'solid',
}

BASE_NUMBER_OF_STEPS = {
    'reacher': 100_000,
    'rccar': 50_000,
    'halfcheetah': 1_000_000,
    'humanoid': 5_000_000,
}

BASE_DISCRETIZATION_STEPS = {
    'Reacher': 0.02,
    'RC Car': 0.5,
    'Halfcheetah': 0.05,
    'Humanoid': 0.015,
}


COLORS = {
    'basline0': 'C0',
    'basline1': 'C1',
    'basline2': 'C2',
    'basline3': 'C3',
    'basline4': 'C4',
}

REVERT_BASELINE_NAMES = {value: key for key, value in BASELINE_NAMES.items()}
LINESTYLES_FROM_NAMES = {BASELINE_NAMES[name]: style for name, style in LINESTYLES.items()}
COLORS_FROM_NAMES = {BASELINE_NAMES[name]: color for name, color in COLORS.items()}

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vu{{\bm{u}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

SWITCH_COST = 0.1  # [0.1, 1, 2, 3]
MAX_TIME_BETWEEN_SWITCHES = 0.5
NUM_EVALS = 10


# We want to add plot
class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_std: np.ndarray
    base_number_of_steps: int = 10


def compute_num_gradient_updates(baseline_name: str, stats: Statistics):
    assert baseline_name in BASELINE_NAMES.keys()
    if baseline_name == 'basline0':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    elif baseline_name == 'basline1':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    elif baseline_name == 'basline2':
        return stats.base_number_of_steps / (stats.xs / stats.xs[-1])
    elif baseline_name == 'basline3':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    else:
        return stats.base_number_of_steps * np.ones_like(stats.xs)


def compute_num_measurements(baseline_name: str, stats: Statistics):
    assert baseline_name in BASELINE_NAMES.keys()
    if baseline_name == 'basline0':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    elif baseline_name == 'basline1':
        return stats.base_number_of_steps / (stats.xs / stats.xs[-1])
    elif baseline_name == 'basline2':
        return stats.base_number_of_steps / (stats.xs / stats.xs[-1])
    elif baseline_name == 'basline3':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    else:
        return stats.base_number_of_steps * np.ones_like(stats.xs)


ENV_NAME_CONVERSION = {
    'reacher': 'Reacher',
    'rccar': 'RC Car',
    'halfcheetah': 'Halfcheetah',
    'humanoid': 'Humanoid'
}

ENV_NAME_CONVERSION_REVERT = {value: key for key, value in ENV_NAME_CONVERSION.items()}


def get_dt(env_name: str):
    if env_name == 'reacher':
        return 0.02
    elif env_name == 'rccar':
        return 0.5
    elif env_name == 'halfcheetah':
        return 0.05
    elif env_name == 'humanoid':
        return 0.015


def compute_physcal_time(baseline_name: str, stats: Statistics):
    assert baseline_name in BASELINE_NAMES.keys()
    if baseline_name == 'basline0':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    elif baseline_name == 'basline1':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    elif baseline_name == 'basline2':
        return stats.base_number_of_steps * np.ones_like(stats.xs)
    elif baseline_name == 'basline3':
        return stats.base_number_of_steps * (stats.xs / stats.xs[-1])
    else:
        return stats.base_number_of_steps * np.ones_like(stats.xs)


systems: Dict[str, Any] = {}

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
        ys_std=np.array(grouped_data['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['rccar']
    )

    cur_data['results/reward_with_switch_cost'] = cur_data['results/total_reward'] - SWITCH_COST * cur_data[
        'results/num_actions']
    grouped_data_with_switch_cost = cur_data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
        ['mean', 'std'])
    grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

    cur_baselines_reward_with_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
        ys_mean=np.array(grouped_data_with_switch_cost['mean']),
        ys_std=np.array(grouped_data_with_switch_cost['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['rccar']
    )
    return cur_baselines_reward_with_switch_cost, cur_baselines_reward_without_switch_cost


baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data_adaptive = pd.read_csv('reward_vs_dt/data/rccar/switch_cost.csv')
filtered_df = data_adaptive[(data_adaptive['switch_cost'] == SWITCH_COST) &
                            (data_adaptive['max_time_between_switches'] == MAX_TIME_BETWEEN_SWITCHES)]
filtered_df['results/reward_with_switch_cost'] = filtered_df['results/total_reward'] - SWITCH_COST * filtered_df[
    'results/num_actions']

data_low_freq = pd.read_csv('reward_vs_dt/data/rccar/low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data_low_freq['results/total_reward'] = data_low_freq['results/total_reward_0']
filtered_df = pd.concat([filtered_df, data_low_freq])

data_equidistant = pd.read_csv('reward_vs_dt/data/rccar/no_switch_cost.csv')
data_equidistant['results/reward_with_switch_cost'] = data_equidistant['results/total_reward'] - SWITCH_COST * \
                                                      data_equidistant['results/num_actions']

data_low_freq_pure_sac = pd.read_csv('reward_vs_dt/data/rccar/low_freq_pure_sac.csv')
data_low_freq_pure_sac['new_integration_dt'] = data_low_freq_pure_sac['new_integration_dt'] * data_low_freq_pure_sac['action_repeat']
data_low_freq_pure_sac['same_amount_of_gradient_updates'] = True
data_low_freq_pure_sac['results/total_reward'] = data_low_freq_pure_sac['results/total_reward_0']
data_equidistant = pd.concat([data_equidistant, data_low_freq_pure_sac])



data_same_gd = data_equidistant[data_equidistant['same_amount_of_gradient_updates'] == True]
data_more_gd = data_equidistant[data_equidistant['same_amount_of_gradient_updates'] == False]

data_naive = pd.read_csv('reward_vs_dt/data/rccar/naive_model.csv')
data_naive['results/total_reward'] = data_naive['results/total_reward_0']
data_naive['results/num_actions'] = data_naive['results/num_actions_0']
data_naive['results/reward_with_switch_cost'] = data_naive['results/total_reward_0'] - SWITCH_COST * \
                                                data_naive['results/num_actions_0']

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

# baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
#     cur_data=data_naive,
#     baseline_name=BASELINE_NAMES['basline3'],
#     cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
#     cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

data_adaptive = pd.read_csv('reward_vs_dt/data/rccar/ppo_switch_cost.csv')

data_low_freq = pd.read_csv('reward_vs_dt/data/rccar/ppo_low_freq.csv')
data_low_freq = data_low_freq[data_low_freq['min_time_repeat'].notnull()]
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data_adaptive = pd.concat([data_adaptive, data_low_freq])

for index in range(1):
    data_adaptive[f'results/reward_with_switch_cost_{index}'] = data_adaptive[
                                                                  f'results/total_reward_{index}'] - SWITCH_COST * \
                                                              data_adaptive[f'results/num_actions_{index}']

statistics = 'mean' # Can be median or mean

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
        ys_std=np.array(grouped_data['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['rccar']
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

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_adaptive,
    baseline_name=BASELINE_NAMES['basline4'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)


systems['RC Car'] = baselines_reward_without_switch_cost
#####################################################################################
#####################################################################################
#####################################################################################


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
        ys_std=np.array(grouped_data['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['reacher']
    )

    cur_data['results/reward_with_switch_cost'] = cur_data['results/total_reward'] - SWITCH_COST * cur_data[
        'results/num_actions']
    grouped_data_with_switch_cost = cur_data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
        ['mean', 'std'])
    grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

    cur_baselines_reward_with_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
        ys_mean=np.array(grouped_data_with_switch_cost['mean']),
        ys_std=np.array(grouped_data_with_switch_cost['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['reacher']
    )
    return cur_baselines_reward_with_switch_cost, cur_baselines_reward_without_switch_cost


########################## Reacher ##########################
#############################################################

baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data_adaptive = pd.read_csv('reward_vs_dt/data/reacher/switch_cost.csv')
filtered_df = data_adaptive[data_adaptive['switch_cost'] == SWITCH_COST]
data_low_freq = pd.read_csv('reward_vs_dt/data/reacher/low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
filtered_df = pd.concat([filtered_df, data_low_freq])

for index in range(NUM_EVALS):
    filtered_df[f'results/reward_with_switch_cost_{index}'] = filtered_df[
                                                                  f'results/total_reward_{index}'] - SWITCH_COST * \
                                                              filtered_df[f'results/num_actions_{index}']

data_equidistant = pd.read_csv('reward_vs_dt/data/reacher/no_switch_cost.csv')
data_low_freq_pure_sac = pd.read_csv('reward_vs_dt/data/reacher/low_freq_pure_sac.csv')
data_low_freq_pure_sac['new_integration_dt'] = data_low_freq_pure_sac['new_integration_dt'] * data_low_freq_pure_sac['action_repeat']
data_low_freq_pure_sac['same_amount_of_gradient_updates'] = True

data_equidistant = pd.concat([data_equidistant, data_low_freq_pure_sac])
for index in range(NUM_EVALS):
    data_equidistant[f'results/reward_with_switch_cost_{index}'] = data_equidistant[
                                                                       f'results/total_reward_{index}'] - SWITCH_COST * \
                                                                   data_equidistant[f'results/num_actions_{index}']

data_equidistant_naive = pd.read_csv('reward_vs_dt/data/reacher/naive_model.csv')
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

# baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
#     cur_data=data_equidistant_naive,
#     baseline_name=BASELINE_NAMES['basline3'],
#     cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
#     cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

data_adaptive = pd.read_csv('reward_vs_dt/data/reacher/ppo_switch_cost.csv')
data_low_freq = pd.read_csv('reward_vs_dt/data/reacher/ppo_low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data = pd.concat([data_adaptive, data_low_freq])

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data,
    baseline_name=BASELINE_NAMES['basline4'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)



systems['Reacher'] = baselines_reward_without_switch_cost



########################## Halfcheetah ######################
#############################################################

SWITCH_COST = 2
MAX_TIME_BETWEEN_SWITCHES = 0.05

baselines_reward_without_switch_cost: Dict[str, Statistics] = {}
baselines_reward_with_switch_cost: Dict[str, Statistics] = {}

data = pd.read_csv('reward_vs_dt/data/halfcheetah/equidistant.csv')
data = data[data['new_integration_dt'] >= 0.05 / 30]
data_adaptive = pd.read_csv('reward_vs_dt/data/halfcheetah/adaptive.csv')
filtered_df = data_adaptive[(data_adaptive['switch_cost'] == SWITCH_COST) &
                            (data_adaptive['max_time_between_switches'] == MAX_TIME_BETWEEN_SWITCHES) &
                            (data_adaptive['time_as_part_of_state'] == True)]
filtered_df['results/reward_with_switch_cost'] = filtered_df['results/total_reward'] - SWITCH_COST * filtered_df[
    'results/num_actions']

data_low_freq = pd.read_csv('reward_vs_dt/data/halfcheetah/low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data_low_freq['results/total_reward'] = data_low_freq['results/total_reward_0']
filtered_df = pd.concat([filtered_df, data_low_freq])

########################################################################################
########################################################################################

grouped_data_adaptive = filtered_df.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data_adaptive = grouped_data_adaptive.reset_index()

baselines_reward_without_switch_cost[BASELINE_NAMES['basline0']] = Statistics(
    xs=np.array(grouped_data_adaptive['new_integration_dt']),
    ys_mean=np.array(grouped_data_adaptive['mean']),
    ys_std=np.array(grouped_data_adaptive['std']),
    base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
)

grouped_data_adaptive_with_switch_cost = filtered_df.groupby('new_integration_dt')[
    'results/reward_with_switch_cost'].agg(['mean', 'std'])
grouped_data_adaptive_with_switch_cost = grouped_data_adaptive_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[BASELINE_NAMES['basline0']] = Statistics(
    xs=np.array(grouped_data_adaptive_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_adaptive_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_adaptive_with_switch_cost['std']),
    base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
)

########################################################################################
########################################################################################

grouped_data = data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data = grouped_data.reset_index()

# baselines_reward_without_switch_cost[
#     BASELINE_NAMES['basline3']] = Statistics(
#     xs=np.array(grouped_data['new_integration_dt']),
#     ys_mean=np.array(grouped_data['mean']),
#     ys_std=np.array(grouped_data['std']),
#     base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
# )

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/total_steps']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

# baselines_reward_with_switch_cost[
#     BASELINE_NAMES['basline3']] = Statistics(
#     xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
#     ys_mean=np.array(grouped_data_with_switch_cost['mean']),
#     ys_std=np.array(grouped_data_with_switch_cost['std']),
#     base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
# )

######### Baseline: Same number of episodes, 1 grad update per env step #########
# data = pd.read_csv('data/halfcheetah/same_number_of_episodes.csv')
data = pd.read_csv('reward_vs_dt/data/halfcheetah/no_switch_cost.csv')
data = data[data['same_amount_of_gradient_updates'] == False]

grouped_data = data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data = grouped_data.reset_index()

baselines_reward_without_switch_cost[BASELINE_NAMES['basline2']] = Statistics(
    xs=np.array(grouped_data['new_integration_dt']),
    ys_mean=np.array(grouped_data['mean']),
    ys_std=np.array(grouped_data['std']),
    base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/num_actions']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[BASELINE_NAMES['basline2']] = Statistics(
    xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_with_switch_cost['std']),
    base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
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
        ys_std=np.array(grouped_data['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
    )

    cur_data['results/reward_with_switch_cost'] = cur_data['results/total_reward'] - SWITCH_COST * cur_data[
        'results/num_actions']
    grouped_data_with_switch_cost = cur_data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
        ['mean', 'std'])
    grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

    cur_baselines_reward_with_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
        ys_mean=np.array(grouped_data_with_switch_cost['mean']),
        ys_std=np.array(grouped_data_with_switch_cost['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
    )
    return cur_baselines_reward_with_switch_cost, cur_baselines_reward_without_switch_cost


# data = pd.read_csv('data/halfcheetah/same_number_of_episodes_and_gradients.csv')
data = pd.read_csv('reward_vs_dt/data/halfcheetah/no_switch_cost.csv')
data = data[data['same_amount_of_gradient_updates'] == True]

data_low_freq_pure_sac = pd.read_csv('reward_vs_dt/data/halfcheetah/low_freq_pure_sac.csv')
data_low_freq_pure_sac['new_integration_dt'] = data_low_freq_pure_sac['new_integration_dt'] * data_low_freq_pure_sac['action_repeat']
data_low_freq_pure_sac['same_amount_of_gradient_updates'] = True
data_low_freq_pure_sac['results/total_reward'] = data_low_freq_pure_sac['results/total_reward_0']
data = pd.concat([data, data_low_freq_pure_sac])

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data,
    baseline_name=BASELINE_NAMES['basline1'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost,
)

data_adaptive = pd.read_csv('reward_vs_dt/data/halfcheetah/ppo_switch_cost.csv')
data_adaptive = data_adaptive[data_adaptive['entropy_cost'] == 0.01]
data_adaptive = data_adaptive[data_adaptive['reward_scaling'] == 1]


data_low_freq = pd.read_csv('reward_vs_dt/data/halfcheetah/ppo_low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data_adaptive = pd.concat([data_adaptive, data_low_freq])
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

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_adaptive,
    baseline_name=BASELINE_NAMES['basline4'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost,
)

systems['Halfcheetah'] = baselines_reward_without_switch_cost

#############################################################
#############################################################

########################## Humanoid #########################
#############################################################

SWITCH_COST = 1.0  # [0.1, 1, 2, 3]
MAX_TIME_BETWEEN_SWITCHES = 0.015
NUM_EVALS = 1
MIN_TIME = 0.015 / 10

statistics = 'median'  # Can be median or mean

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
        ys_std=np.array(grouped_data['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['humanoid']
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

data_adaptive = pd.read_csv('reward_vs_dt/data/humanoid/switch_cost.csv')
data_adaptive = data_adaptive[data_adaptive['new_integration_dt'] >= MIN_TIME]
filtered_df = data_adaptive[data_adaptive['switch_cost'] == SWITCH_COST]

data_low_freq = pd.read_csv('reward_vs_dt/data/humanoid/low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
filtered_df = pd.concat([filtered_df, data_low_freq])

for index in range(NUM_EVALS):
    filtered_df[f'results/reward_with_switch_cost_{index}'] = filtered_df[
                                                                  f'results/total_reward_{index}'] - SWITCH_COST * \
                                                              filtered_df[f'results/num_actions_{index}']

data_equidistant = pd.read_csv('reward_vs_dt/data/humanoid/no_switch_cost.csv')
data_equidistant = data_equidistant[data_equidistant['new_integration_dt'] >= MIN_TIME]
data_low_freq_pure_sac = pd.read_csv('reward_vs_dt/data/humanoid/low_freq_pure_sac.csv')
data_low_freq_pure_sac['new_integration_dt'] = data_low_freq_pure_sac['new_integration_dt'] * data_low_freq_pure_sac['action_repeat']
data_low_freq_pure_sac['same_amount_of_gradient_updates'] = True

data_equidistant = pd.concat([data_equidistant, data_low_freq_pure_sac])
for index in range(NUM_EVALS):
    data_equidistant[f'results/reward_with_switch_cost_{index}'] = data_equidistant[
                                                                       f'results/total_reward_{index}'] - SWITCH_COST * \
                                                                   data_equidistant[f'results/num_actions_{index}']

data_equidistant_naive = pd.read_csv('reward_vs_dt/data/humanoid/naive_model.csv')
data_equidistant_naive = data_equidistant_naive[data_equidistant_naive['new_integration_dt'] >= MIN_TIME]

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

# baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
#     cur_data=data_equidistant_naive,
#     baseline_name=BASELINE_NAMES['basline3'],
#     cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
#     cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)


statistics = 'mean' # Can be median or mean


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
        ys_std=np.array(grouped_data['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['humanoid']
    )

    cur_data['results/reward_with_switch_cost'] = cur_data['results/total_reward'] - SWITCH_COST * cur_data[
        'results/num_actions']
    grouped_data_with_switch_cost = cur_data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
        [statistics, 'std'])
    grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

    cur_baselines_reward_with_switch_cost[baseline_name] = Statistics(
        xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
        ys_mean=np.array(grouped_data_with_switch_cost[statistics]),
        ys_std=np.array(grouped_data_with_switch_cost['std']),
        base_number_of_steps=BASE_NUMBER_OF_STEPS['humanoid']
    )
    return cur_baselines_reward_with_switch_cost, cur_baselines_reward_without_switch_cost


data_adaptive = pd.read_csv('reward_vs_dt/data/humanoid/ppo_switch_cost.csv')
data_low_freq = pd.read_csv('reward_vs_dt/data/humanoid/ppo_low_freq.csv')
data_low_freq['new_integration_dt'] = data_low_freq['new_integration_dt'] * data_low_freq['min_time_repeat']
data_adaptive = pd.concat([data_adaptive, data_low_freq])

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_adaptive,
    baseline_name=BASELINE_NAMES['basline4'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)


systems['Humanoid'] = baselines_reward_without_switch_cost

################################################################################################
################################################################################################
############################ Slices from Training #######################################
################################################################################################
################################################################################################
################################################################################################


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

from typing import List

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

data_path = f'learning_curves/data/{env}/high_freq_sac_tac.csv'
data = pd.read_csv(data_path)
data = data[data['switch_cost'] == 1]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline0'],
    statistics=stats,
    add_transition_cost=True
)

data_path = f'learning_curves/data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == True]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline1'],
    statistics=stats,
    add_transition_cost=False
)

data_path = f'learning_curves/data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == False]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline2'],
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

data_path = 'learning_curves/data/reacher/high_freq_sac_tac.csv'
data = pd.read_csv(data_path)
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline0'],
    statistics=stats,
    add_transition_cost=True
)

data_path = 'learning_curves/data/reacher/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == True]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline1'],
    statistics=stats,
    add_transition_cost=False
)

data_path = 'learning_curves/data/reacher/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == False]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline2'],
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

data_path = f'learning_curves/data/{env}/high_freq_sac_tac.csv'
data = pd.read_csv(data_path)
data = data[data['switch_cost'] == transition_cost]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline0'],
    statistics=stats,
    add_transition_cost=True
)

data_path = f'learning_curves/data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == True]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline1'],
    statistics=stats,
    add_transition_cost=False
)

data_path = f'learning_curves/data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == False]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline2'],
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

data_path = f'learning_curves/data/{env}/high_freq_sac_tac.csv'
data = pd.read_csv(data_path)
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline0'],
    statistics=stats,
    add_transition_cost=True
)

data_path = f'learning_curves/data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == True]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline1'],
    statistics=stats,
    add_transition_cost=False
)

data_path = f'learning_curves/data/{env}/high_freq_sac.csv'
data = pd.read_csv(data_path)
data = data[data['same_amount_of_gradient_updates'] == False]
stats = update_statistics(
    data=data,
    baseline_name=BASELINE_NAMES['basline2'],
    statistics=stats,
    add_transition_cost=False
)

systems_eval['Humanoid'] = stats


################################################################################################
################################################################################################
############################ Plotting #######################################
################################################################################################
################################################################################################
################################################################################################


fig = plt.figure(figsize=(20, 8))
handles, labels = [], []

gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])

init_row = []

TRAINING_SNAPS = [80, 10, 10, 5]

for index, (title, baselines) in enumerate(systems.items()):
    ax = fig.add_subplot(gs[0, index])
    init_row.append(ax)
    ax.axvline(x=1 / BASE_DISCRETIZATION_STEPS[title],
               color='black',
               ls='--',
               alpha=0.4,
               linewidth=LINE_WIDTH,
               label=r'1/$t^*$')
    ax.axvline(x=1 / BASE_DISCRETIZATION_STEPS[title] * TRAINING_SNAPS[index],
               color='red',
               ls='--',
               # alpha=0.4,
               linewidth=LINE_WIDTH,
               label=r'1/$t_{\text{eval}}$')
    for baseline_name, baseline_stat in baselines.items():
        ax.plot(1 / baseline_stat.xs, baseline_stat.ys_mean,
                label=baseline_name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[baseline_name],
                color=COLORS_FROM_NAMES[baseline_name], )
        ax.fill_between(1 / baseline_stat.xs,
                        baseline_stat.ys_mean - baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                        baseline_stat.ys_mean + baseline_stat.ys_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                        color=COLORS_FROM_NAMES[baseline_name],
                        alpha=0.2)

    ax.set_xscale('log')
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=50)
    if index == 0:
        ax.set_ylabel(r'Episode Reward', fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel(r'1/$t_{\min}$', fontsize=LABEL_FONT_SIZE)
    # plt.setp(ax.get_xticklabels(), visible=False)


    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)


for index, (system, stats) in enumerate(systems_eval.items()):
    ax = fig.add_subplot(gs[1, index])
    ax.set_xscale('linear')
    for stat in stats:
        ax.plot(stat.xs, stat.ys_mean,
                label=stat.name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[stat.name],
                color=COLORS_FROM_NAMES[stat.name],
                )
        ax.fill_between(
            stat.xs,
            stat.ys_mean - stat.ys_std / np.sqrt(NUM_SAMPLES),
            stat.ys_mean + stat.ys_std / np.sqrt(NUM_SAMPLES),
            color=COLORS_FROM_NAMES[stat.name],
            alpha=0.2,
        )
    if system == 'Humanoid':
        ax.set_xlim(0, 100_000)
    ax.set_xlabel('Physical Time [sec]', fontsize=LABEL_FONT_SIZE)
    if index == 0:
        ax.set_ylabel('Episode reward', fontsize=LABEL_FONT_SIZE)
    # ax.set_title(system, fontsize=TITLE_FONT_SIZE)

    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=6,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.94),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.tight_layout(rect=[0.0, 0.0, 1, 0.98])
plt.savefig('varying_integration_dt.pdf')
plt.show()
