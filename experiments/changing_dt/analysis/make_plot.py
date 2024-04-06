import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

LEGEND_FONT_SIZE = 26
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

SWITCH_COST = 1.0

data = pd.read_csv('data/halfcheetah/equidistant.csv')
data_adaptive = pd.read_csv('data/halfcheetah/adaptive.csv')
filtered_df = data_adaptive[(data_adaptive['switch_cost'] == SWITCH_COST) &
                            (data_adaptive['max_time_between_switches'] == 0.05) &
                            (data_adaptive['time_as_part_of_state'] == True)]
filtered_df['results/reward_with_switch_cost'] = filtered_df['results/total_reward'] - SWITCH_COST * filtered_df['results/num_actions']
########################################################################################
########################################################################################

grouped_data_adaptive = filtered_df.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data_adaptive = grouped_data_adaptive.reset_index()

xs_grouped_data_adaptive = np.array(grouped_data_adaptive['new_integration_dt'])
ys_grouped_data_mean_adaptive = np.array(grouped_data_adaptive['mean'])
ys_grouped_data_std_adaptive = np.array(grouped_data_adaptive['std'])

grouped_data_adaptive_with_switch_cost = filtered_df.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(['mean', 'std'])
grouped_data_adaptive_with_switch_cost = grouped_data_adaptive_with_switch_cost.reset_index()

xs_grouped_data_adaptive_with_switch_cost = np.array(grouped_data_adaptive_with_switch_cost['new_integration_dt'])
ys_grouped_data_mean_adaptive_with_switch_cost = np.array(grouped_data_adaptive_with_switch_cost['mean'])
ys_grouped_data_std_adaptive_with_switch_cost = np.array(grouped_data_adaptive_with_switch_cost['std'])

########################################################################################
########################################################################################

grouped_data = data.groupby('new_integration_dt')['results/total_reward'].agg(['mean', 'std'])
grouped_data = grouped_data.reset_index()

xs_grouped_data = np.array(grouped_data['new_integration_dt'])
ys_grouped_data_mean = np.array(grouped_data['mean'])
ys_grouped_data_std = np.array(grouped_data['std'])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

ax[0].plot(xs_grouped_data, ys_grouped_data_mean, label='Control per integration step')
ax[0].fill_between(xs_grouped_data,
                   ys_grouped_data_mean - ys_grouped_data_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   ys_grouped_data_mean + ys_grouped_data_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   alpha=0.2)

ax[0].plot(xs_grouped_data_adaptive, ys_grouped_data_mean_adaptive, label='Switch cost control')
ax[0].fill_between(xs_grouped_data_adaptive,
                   ys_grouped_data_mean_adaptive - ys_grouped_data_std_adaptive / np.sqrt(NUM_SAMPLES_PER_SEED),
                   ys_grouped_data_mean_adaptive + ys_grouped_data_std_adaptive / np.sqrt(NUM_SAMPLES_PER_SEED),
                   alpha=0.2)

ax[0].set_xlabel(r'Integration dt', fontsize=LABEL_FONT_SIZE)
ax[0].set_ylabel('Reward [Without Switch Cost]', fontsize=LABEL_FONT_SIZE)

data['results/reward_with_switch_cost'] = data['results/total_reward'] - SWITCH_COST * data['results/total_steps']
grouped_data_with_switch_cost = data.groupby('new_integration_dt')['results/reward_with_switch_cost'].agg(
    ['mean', 'std'])
grouped_data_with_switch_cost = grouped_data_with_switch_cost.reset_index()

xs_grouped_data_with_switch_cost = np.array(grouped_data_with_switch_cost['new_integration_dt'])
ys_grouped_data_mean_with_switch_cost = np.array(grouped_data_with_switch_cost['mean'])
ys_grouped_data_std_with_switch_cost = np.array(grouped_data_with_switch_cost['std'])

ax[1].plot(xs_grouped_data_with_switch_cost, ys_grouped_data_mean_with_switch_cost, label='Control per integration step')
ax[1].fill_between(xs_grouped_data,
                   ys_grouped_data_mean_with_switch_cost - ys_grouped_data_std_with_switch_cost / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   ys_grouped_data_mean_with_switch_cost + ys_grouped_data_std_with_switch_cost / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   alpha=0.2)

ax[1].plot(xs_grouped_data_adaptive_with_switch_cost, ys_grouped_data_mean_adaptive_with_switch_cost, label='Switch cost control')
ax[1].fill_between(xs_grouped_data_adaptive_with_switch_cost,
                   ys_grouped_data_mean_adaptive_with_switch_cost - ys_grouped_data_std_adaptive_with_switch_cost / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   ys_grouped_data_mean_adaptive_with_switch_cost + ys_grouped_data_std_adaptive_with_switch_cost / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   alpha=0.2)


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

fig.suptitle('Halfcheetah run forward task [Duration = 10 sec], [Switch Cost = 1.0]',
             fontsize=TITLE_FONT_SIZE,
             y=0.92)
fig.tight_layout(rect=[0.0, 0.0, 1, 0.88])
plt.savefig('halfcheetah_switch_cost_varying_integration_dt.pdf')
plt.show()
