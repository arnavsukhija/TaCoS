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
    'basline0': 'SAC-WTC',
    'basline1': 'SAC-MS',
    'basline2': 'SAC-MCMS',
    'basline3': 'SAC',
    'basline4': 'PPO-WTC',
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

data_equidistant_naive = pd.read_csv('data/reacher/naive_model.csv')
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

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_equidistant_naive,
    baseline_name=BASELINE_NAMES['basline3'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

data_adaptive = pd.read_csv('data/reacher/ppo_switch_cost.csv')

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_adaptive,
    baseline_name=BASELINE_NAMES['basline4'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)



systems['Reacher'] = baselines_reward_without_switch_cost


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

data_naive = pd.read_csv('data/rccar/naive_model.csv')
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

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_naive,
    baseline_name=BASELINE_NAMES['basline3'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)

data_adaptive = pd.read_csv('data/rccar/ppo_switch_cost.csv')

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

baselines_reward_without_switch_cost[
    BASELINE_NAMES['basline3']] = Statistics(
    xs=np.array(grouped_data['new_integration_dt']),
    ys_mean=np.array(grouped_data['mean']),
    ys_std=np.array(grouped_data['std']),
    base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/total_steps']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

baselines_reward_with_switch_cost[
    BASELINE_NAMES['basline3']] = Statistics(
    xs=np.array(grouped_data_with_switch_cost['new_integration_dt']),
    ys_mean=np.array(grouped_data_with_switch_cost['mean']),
    ys_std=np.array(grouped_data_with_switch_cost['std']),
    base_number_of_steps=BASE_NUMBER_OF_STEPS['halfcheetah']
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
data = pd.read_csv('data/halfcheetah/no_switch_cost.csv')
data = data[data['same_amount_of_gradient_updates'] == True]

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data,
    baseline_name=BASELINE_NAMES['basline1'],
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

data_adaptive = pd.read_csv('data/humanoid/switch_cost.csv')
data_adaptive = data_adaptive[data_adaptive['new_integration_dt'] >= MIN_TIME]
filtered_df = data_adaptive[data_adaptive['switch_cost'] == SWITCH_COST]

for index in range(NUM_EVALS):
    filtered_df[f'results/reward_with_switch_cost_{index}'] = filtered_df[
                                                                  f'results/total_reward_{index}'] - SWITCH_COST * \
                                                              filtered_df[f'results/num_actions_{index}']

data_equidistant = pd.read_csv('data/humanoid/no_switch_cost.csv')
data_equidistant = data_equidistant[data_equidistant['new_integration_dt'] >= MIN_TIME]

for index in range(NUM_EVALS):
    data_equidistant[f'results/reward_with_switch_cost_{index}'] = data_equidistant[
                                                                       f'results/total_reward_{index}'] - SWITCH_COST * \
                                                                   data_equidistant[f'results/num_actions_{index}']

data_equidistant_naive = pd.read_csv('data/humanoid/naive_model.csv')
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

baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_equidistant_naive,
    baseline_name=BASELINE_NAMES['basline3'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)


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


data_adaptive = pd.read_csv('data/humanoid/ppo_switch_cost.csv')


baselines_reward_with_switch_cost, baselines_reward_without_switch_cost = update_baselines(
    cur_data=data_adaptive,
    baseline_name=BASELINE_NAMES['basline4'],
    cur_baselines_reward_with_switch_cost=baselines_reward_with_switch_cost,
    cur_baselines_reward_without_switch_cost=baselines_reward_without_switch_cost)


systems['Humanoid'] = baselines_reward_without_switch_cost

#############################################################
#############################################################

fig = plt.figure(figsize=(20, 14))
handles, labels = [], []

gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1])

init_row = []

for index, (title, baselines) in enumerate(systems.items()):
    ax = fig.add_subplot(gs[0, index])
    init_row.append(ax)
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
        ax.set_ylabel(r'Reward [No $c(\vx, \vu, t)$]', fontsize=LABEL_FONT_SIZE)

    plt.setp(ax.get_xticklabels(), visible=False)


    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

for index, (title, baselines) in enumerate(systems.items()):
    ax = fig.add_subplot(gs[1, index], sharex=init_row[index])
    for baseline_name, baseline_stat in baselines.items():
        print(f'Title: {title}, baseline: {baseline_name}')
        print(compute_num_measurements(REVERT_BASELINE_NAMES[baseline_name], baseline_stat))
        ax.plot(1 / baseline_stat.xs,
                compute_num_measurements(REVERT_BASELINE_NAMES[baseline_name], baseline_stat),
                label=baseline_name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[baseline_name],
                color=COLORS_FROM_NAMES[baseline_name], )

    ax.set_xscale('log')
    ax.set_yscale('log')
    if index == 0:
        ax.set_ylabel(r'\# Samples', fontsize=LABEL_FONT_SIZE)

    plt.setp(ax.get_xticklabels(), visible=False)

    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

for index, (title, baselines) in enumerate(systems.items()):
    ax = fig.add_subplot(gs[2, index], sharex=init_row[index])
    for baseline_name, baseline_stat in baselines.items():
        ax.plot(1 / baseline_stat.xs,
                compute_num_gradient_updates(REVERT_BASELINE_NAMES[baseline_name], baseline_stat),
                label=baseline_name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[baseline_name],
                color=COLORS_FROM_NAMES[baseline_name], )

    ax.set_xscale('log')
    ax.set_yscale('log')
    if index == 0:
        ax.set_ylabel(r'Computation', fontsize=LABEL_FONT_SIZE)
    plt.setp(ax.get_xticklabels(), visible=False)


    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

for index, (title, baselines) in enumerate(systems.items()):
    ax = fig.add_subplot(gs[3, index], sharex=init_row[index])
    for baseline_name, baseline_stat in baselines.items():
        ax.plot(1 / baseline_stat.xs,
                compute_physcal_time(REVERT_BASELINE_NAMES[baseline_name], baseline_stat) * get_dt(
                    ENV_NAME_CONVERSION_REVERT[title]),
                label=baseline_name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[baseline_name],
                color=COLORS_FROM_NAMES[baseline_name], )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Frequency', fontsize=LABEL_FONT_SIZE)
    if index == 0:
        ax.set_ylabel(r'Env Time [sec]', fontsize=LABEL_FONT_SIZE)

    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=5,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.98),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.tight_layout()
plt.savefig('varying_integration_dt.pdf')
plt.show()
