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

data = pd.read_csv('data/greenhouse_data.csv')
data['results/reward_with_switch_cost'] = data['results/total_reward'] - 0.2 * data['results/num_actions']

data_performance = pd.read_csv('data/noise_influence_performance.csv')
data_performance = data_performance[data_performance['env_name'] == 'Greenhouse']

# # Prepare data for number of actions
grouped_num_actions = data.groupby('scale')['results/num_actions'].agg(['mean', 'std'])
grouped_num_actions = grouped_num_actions.reset_index()

xs_num_actions = np.array(grouped_num_actions['scale'])
ys_num_actions_mean = np.array(grouped_num_actions['mean'])
ys_num_actions_std = np.array(grouped_num_actions['std'])

# Smooth to get the trend
sigma = 3  # Standard deviation for Gaussian kernel
ys_num_actions_mean = gaussian_filter1d(ys_num_actions_mean, sigma)
ys_num_actions_std = gaussian_filter1d(ys_num_actions_std, sigma)

# # Prepare data for achieved reward
grouped_total_reward = data.groupby('scale')['results/total_reward'].agg(['mean', 'std'])
grouped_total_reward = grouped_total_reward.reset_index()

grouped_total_reward_full_control = data_performance.groupby('scale')['results/total_reward'].agg(['mean', 'std'])
grouped_total_reward_full_control = grouped_total_reward_full_control.reset_index()

xs_total_reward = np.array(grouped_total_reward['scale'])
ys_total_rewards_mean = np.array(grouped_total_reward['mean'])
ys_total_reward_std = np.array(grouped_total_reward['std'])

xs_total_reward_full_control = np.array(grouped_total_reward_full_control['scale'])
ys_total_rewards_mean_full_control = np.array(grouped_total_reward_full_control['mean'])
ys_total_reward_std_full_control = np.array(grouped_total_reward_full_control['std'])

# Prepare data for the reward with switch cost
grouped_reward_with_switch_cost = data.groupby('scale')['results/reward_with_switch_cost'].agg(['mean', 'std'])
grouped_reward_with_switch_cost = grouped_reward_with_switch_cost.reset_index()

xs_reward_with_switch_cost = np.array(grouped_reward_with_switch_cost['scale'])
ys_reward_with_switch_cost_mean = np.array(grouped_reward_with_switch_cost['mean'])
ys_reward_with_switch_cost_std = np.array(grouped_reward_with_switch_cost['std'])

# Smooth to get the trend
sigma = 3  # Standard deviation for Gaussian kernel
ys_total_rewards_mean_full_control = gaussian_filter1d(ys_total_rewards_mean_full_control, sigma)
ys_total_reward_std_full_control = gaussian_filter1d(ys_total_reward_std_full_control, sigma)


# Smooth to get the trend
sigma = 3  # Standard deviation for Gaussian kernel
ys_reward_with_switch_cost_mean = gaussian_filter1d(ys_reward_with_switch_cost_mean, sigma)
ys_reward_with_switch_cost_std = gaussian_filter1d(ys_reward_with_switch_cost_std, sigma)

# Smooth to get the trend
sigma = 3  # Standard deviation for Gaussian kernel
ys_total_rewards_mean = gaussian_filter1d(ys_total_rewards_mean, sigma)
ys_total_reward_std = gaussian_filter1d(ys_total_reward_std, sigma)

# Plotting
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

ax[0].plot(xs_num_actions, ys_num_actions_mean)
ax[0].fill_between(xs_num_actions,
                   ys_num_actions_mean - ys_num_actions_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   ys_num_actions_mean + ys_num_actions_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   alpha=0.2)
ax[0].set_xlabel(r'Noise magnitude: $x \times $ DEFAULT_STD', fontsize=LABEL_FONT_SIZE)
ax[0].set_ylabel('Number of applied actions', fontsize=LABEL_FONT_SIZE)

ax[1].plot(xs_num_actions, ys_total_rewards_mean, label='Switch cost control')
ax[1].fill_between(xs_total_reward,
                   ys_total_rewards_mean - ys_total_reward_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   ys_total_rewards_mean + ys_total_reward_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   alpha=0.2)

ax[1].plot(xs_total_reward_full_control, ys_total_rewards_mean_full_control,
           label="Control per integration step [200 actions]")
ax[1].fill_between(xs_total_reward_full_control,
                   ys_total_rewards_mean_full_control - ys_total_reward_std_full_control / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   ys_total_rewards_mean_full_control + ys_total_reward_std_full_control / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   alpha=0.2)
# ax[1].legend(fontsize=LEGEND_FONT_SIZE, loc='lower left')
ax[1].set_xlabel(r'Noise magnitude: $x \times $ DEFAULT_STD', fontsize=LABEL_FONT_SIZE)
ax[1].set_ylabel('Reward [Without Switch Cost]', fontsize=LABEL_FONT_SIZE)


ax[2].plot(xs_num_actions, ys_reward_with_switch_cost_mean, label='Switch cost control')
ax[2].fill_between(xs_total_reward,
                   ys_reward_with_switch_cost_mean - ys_reward_with_switch_cost_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   ys_reward_with_switch_cost_mean + ys_reward_with_switch_cost_std / np.sqrt(NUM_SAMPLES_PER_SEED),
                   alpha=0.2)

ax[2].plot(xs_total_reward_full_control, ys_total_rewards_mean_full_control - 40,
           label="Control per integration step [200 actions]")
ax[2].fill_between(xs_total_reward_full_control,
                   ys_total_rewards_mean_full_control - 40 - ys_total_reward_std_full_control / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   ys_total_rewards_mean_full_control - 40 + ys_total_reward_std_full_control / np.sqrt(
                       NUM_SAMPLES_PER_SEED),
                   alpha=0.2)
# ax[2].legend(fontsize=LEGEND_FONT_SIZE, loc='lower left')
ax[2].set_xlabel(r'Noise magnitude: $x \times $ DEFAULT_STD', fontsize=LABEL_FONT_SIZE)
ax[2].set_ylabel('Reward [With Switch Cost]', fontsize=LABEL_FONT_SIZE)

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

fig.suptitle('Greenhouse temperature-tracking task [Duration = 200 min], [Switch Cost = 0.2]',
             fontsize=TITLE_FONT_SIZE,
             y=0.92)
fig.tight_layout(rect=[0.0, 0.0, 1, 0.88])
plt.savefig('greenhouse_switch_cost_varying_noise.pdf')
plt.show()
