import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

LEGEND_FONT_SIZE = 26
TITLE_FONT_SIZE = 26
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

EPISODE_LEN = 300
NUMBER_OF_SAMPLES = 5

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

data = pd.read_csv('data/greenhouse_bounded_switches.csv')
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

# Plotting
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

ax.plot(xs_bounded_switches, ys_bounded_switches_mean, label='Optimized time between actions')
ax.fill_between(xs_bounded_switches,
                ys_bounded_switches_mean - 2 * ys_bounded_switches_std / np.sqrt(NUMBER_OF_SAMPLES),
                ys_bounded_switches_mean + 2 * ys_bounded_switches_std / np.sqrt(NUMBER_OF_SAMPLES),
                alpha=0.2)

ax.plot(xs_repeated_actions, ys_repeated_actions_mean, label='Equidistant time between actions')
ax.fill_between(xs_repeated_actions,
                ys_repeated_actions_mean - 2 * ys_repeated_actions_std / np.sqrt(NUMBER_OF_SAMPLES),
                ys_repeated_actions_mean + 2 * ys_repeated_actions_std / np.sqrt(NUMBER_OF_SAMPLES),
                alpha=0.2)

ax.set_xlim(0, 35)
ax.set_xlabel('Number of applied actions', fontsize=LABEL_FONT_SIZE)
ax.set_ylabel('Total reward', fontsize=LABEL_FONT_SIZE)
ax.legend(fontsize=LEGEND_FONT_SIZE, loc='lower right')
ax.set_title('Greenhouse Temperature Tracking [Duration=25h]',
             fontsize=TITLE_FONT_SIZE)
plt.tight_layout()
plt.savefig('greenhouse_temperature_tracking.pdf')
plt.show()
