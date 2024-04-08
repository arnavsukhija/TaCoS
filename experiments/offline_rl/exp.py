"""
Simple offline RL experiment on the Pendulum System
"""

import datetime
from datetime import datetime

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from brax.envs.base import PipelineEnv, State, Env
from bsm.bayesian_regression import DeterministicEnsemble
from bsm.statistical_model import BNNStatisticalModel
from bsm.utils.normalization import Data
from jax import vmap
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

import wandb
from wtc.envs.pendulum import PendulumEnv
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost, SwitchCost

env = PendulumEnv(reward_source='dm-control')

episode_len = 100
episode_time = episode_len * env.dt

min_time_between_switches = 1 * env.dt
max_time_between_switches = 30 * env.dt

# Wrap env with IHSwitchCostWrapper
env = IHSwitchCostWrapper(env,
                          num_integrator_steps=episode_len,
                          min_time_between_switches=1 * env.dt,
                          max_time_between_switches=30 * env.dt,
                          switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                          time_as_part_of_state=False,
                          discounting=1.0
                          )

# Prepare input data
number_offline_data = 200
seed = 0

key = jr.PRNGKey(seed)

# Sample actions
key, key_actions = jr.split(key)
actions = jr.uniform(key=key_actions, shape=(number_offline_data, env.action_size), minval=-1, maxval=1)

# Sample states
key, key_angle, key_angular_velocity = jr.split(key, 3)
angle = jr.uniform(key=key_angle, shape=(number_offline_data, 1), minval=-jnp.pi, maxval=jnp.pi)
velocity = jr.uniform(key=key_angular_velocity, shape=(number_offline_data, 1), minval=-5, maxval=5)

# Transform data to cos ant sin
cos, sin = jnp.cos(angle), jnp.sin(angle)
obs = jnp.concatenate([cos, sin, velocity], axis=-1)

key, key_init = jr.split(key)
key_init = jr.split(key_init, number_offline_data)
init_state = vmap(env.reset)(key_init)
init_state = init_state.replace(obs=obs)
next_state = vmap(env.step)(init_state, actions)

inputs = jnp.concatenate([init_state.obs, actions], axis=-1)
outputs = jnp.concatenate([next_state.obs, next_state.reward.reshape(-1, 1)], axis=-1)

data = Data(inputs=inputs, outputs=outputs)

wandb.init(
    project='OfflineRL'
)

model = BNNStatisticalModel(
    input_dim=env.observation_size + env.action_size,
    output_dim=env.observation_size + 1,  # One more for the reward
    output_stds=0.1 * jnp.ones(shape=(env.observation_size + 1,)),
    logging_wandb=True,
    beta=1.0 * jnp.ones(shape=(env.observation_size + 1,)),
    num_particles=5,
    features=(128,) * 5,
    bnn_type=DeterministicEnsemble,
    batch_size=128,
    train_share=0.8,
    num_training_steps=2_000,
    eval_frequency=100,
    weight_decay=0.0,
    return_best_model=True,
)

key, key_training = jr.split(key)
init_stats_model_state = model.init(key=key_training)
statistical_model_state = model.update(stats_model_state=init_stats_model_state, data=data)


class PendulumTrainedEnv(Env):
    def __init__(self,
                 env: PipelineEnv,
                 num_integrator_steps: int,
                 min_time_between_switches: float,
                 max_time_between_switches: float | None = None,
                 switch_cost: SwitchCost = ConstantSwitchCost(value=jnp.array(1.0)),
                 discounting: float = 0.99,
                 model: DeterministicEnsemble = None,
                 model_state: DeterministicEnsemble = None
                 ):
        self.env = env
        self.num_integrator_steps = num_integrator_steps
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches
        assert min_time_between_switches >= self.env.dt, \
            'Min time between switches must be at least of the integration time dt'
        self.time_horizon = self.env.dt * self.num_integrator_steps
        if max_time_between_switches is None:
            max_time_between_switches = self.time_horizon
        self.max_time_between_switches = max_time_between_switches
        self.discounting = discounting
        self.model = model
        self.model_state = model_state

    def reset(self, rng: jax.Array) -> State:
        """
        The augmented state is represented by concatenated vector:
         (state, time-to-go)
        """
        state = self.env.reset(rng)
        time = jnp.array(0.0)
        augmented_obs = jnp.concatenate([state.obs, time.reshape(1)])
        augmented_state = state.replace(obs=augmented_obs)
        return augmented_state

    def compute_time(self,
                     pseudo_time: chex.Array,
                     dt: chex.Array,
                     t_lower: chex.Array,
                     t_upper: chex.Array,
                     ) -> chex.Array:
        time_for_action = ((t_upper - t_lower) / 2 * pseudo_time + (t_upper + t_lower) / 2)
        return (time_for_action // dt) * dt

    def step(self, state: State, action: jax.Array) -> State:
        u, pseudo_time_for_action = action[:-1], action[-1]
        obs, time = state.obs[:-1], state.obs[-1]
        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, self.max_time_between_switches]
        time_for_action = self.compute_time(pseudo_time=pseudo_time_for_action,
                                            dt=self.env.dt,
                                            t_lower=self.min_time_between_switches,
                                            t_upper=self.max_time_between_switches,
                                            )
        done = time_for_action >= self.time_horizon - time
        done = 1 - (1 - done) * (1 - state.done)

        # Integrate dynamics forward for the num_steps
        model_input = jnp.concatenate([obs, action])
        model_output = self.model(model_input, self.model_state)
        mean_pred = model_output.mean
        next_obs = mean_pred[:-1]
        th = jnp.arctan2(next_obs[1], next_obs[0])
        next_obs = jnp.asarray([jnp.cos(th), jnp.sin(th), next_obs[2]]).reshape(-1)
        reward = mean_pred[-1]

        # Add switch cost to the total reward
        total_reward = reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs
        next_time = (time + time_for_action).reshape(1)

        augmented_next_obs = jnp.concatenate([next_obs, next_time])
        augmented_next_state = state.replace(obs=augmented_next_obs,
                                             reward=total_reward,
                                             done=done)
        return augmented_next_state

    @property
    def observation_size(self) -> int:
        # +1 for time-to-go ant +1 for num remaining switches
        return self.env.observation_size + 1

    @property
    def action_size(self) -> int:
        # +1 for time that we apply action for
        return self.env.action_size + 1

    @property
    def backend(self) -> str:
        return self.env.backend

    @property
    def dt(self):
        return self.env.dt


env = PendulumEnv(reward_source='dm-control')

env = PendulumTrainedEnv(
    env=env,
    num_integrator_steps=episode_len,
    min_time_between_switches=1 * env.dt,
    max_time_between_switches=30 * env.dt,
    switch_cost=ConstantSwitchCost(value=jnp.array(0.1)),
    discounting=1.0,
    model=model,
    model_state=statistical_model_state,
)

discount_factor = 0.99
continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                            dt=env.dt)
num_env_steps_between_updates = 5
num_envs = 32
optimizer = SAC(
    target_entropy=None,
    environment=env,
    num_timesteps=20_000,
    episode_length=episode_len,
    action_repeat=1,
    num_env_steps_between_updates=num_env_steps_between_updates,
    num_envs=num_envs,
    num_eval_envs=32,
    lr_alpha=3e-4,
    lr_policy=3e-4,
    lr_q=3e-4,
    wd_alpha=0.,
    wd_policy=0.,
    wd_q=0.,
    max_grad_norm=1e5,
    discounting=0.99,
    batch_size=64,
    num_evals=20,
    normalize_observations=True,
    reward_scaling=1.,
    tau=0.005,
    min_replay_size=10 ** 3,
    max_replay_size=10 ** 5,
    grad_updates_per_step=num_env_steps_between_updates * num_envs,
    deterministic_eval=True,
    init_log_alpha=0.,
    policy_hidden_layer_sizes=(32,) * 5,
    policy_activation=swish,
    critic_hidden_layer_sizes=(128,) * 3,
    critic_activation=swish,
    wandb_logging=True,
    return_best_model=True,
    non_equidistant_time=True,
    continuous_discounting=continuous_discounting,
    min_time_between_switches=min_time_between_switches,
    max_time_between_switches=max_time_between_switches,
    env_dt=env.dt,
)

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    # plt.xlim([0, train_fn.keywords['num_timesteps']])
    # plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


print('Before inference')
# wandb.init(
#     project='TestIHSwitchCost'
# )
policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(0), progress_fn=progress)
print('After inference')

# Now we plot the evolution
pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)


def policy(obs):
    return pseudo_policy(obs, key_sample=jr.PRNGKey(0))


# Evaluate on the true Pendulum system

env = PendulumEnv(reward_source='dm-control',
                  add_process_noise=True,
                  process_noise_scale=0.0 * jnp.array([0.01, 0.01, 0.1]))

env = IHSwitchCostWrapper(env,
                          num_integrator_steps=episode_len,
                          min_time_between_switches=1 * env.dt,
                          max_time_between_switches=30 * env.dt,
                          switch_cost=ConstantSwitchCost(value=jnp.array(0.1)),
                          time_as_part_of_state=True,
                          discounting=discount_factor)


# @jax.jit
def step(state, _):
    u = policy(state.obs)[0]
    print('Step')
    print(f'Time to go {u[-1]}')
    next_state, rest = env.simulation_step(state, u)
    return next_state, (next_state.obs, u, next_state.reward, rest)


state = env.reset(rng=jr.PRNGKey(0))

PLOT_TRUE_TRAJECTORIES = True
time_as_part_of_state = True

init_state = state
LEGEND_SIZE = 20
LABEL_SIZE = 20
TICKS_SIZE = 20

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

trajectory = []
full_trajectories = []
all_states = []
while not state.done:
    state, one_traj = step(state, None)
    one_traj, full_trajectory = one_traj[:-1], one_traj[-1]
    trajectory.append(one_traj)
    full_trajectories.append(full_trajectory)
    all_states.append(state)

trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)
full_trajectory = jtu.tree_map(lambda *xs: jnp.concatenate(xs), *full_trajectories)
all_states = jtu.tree_map(lambda *xs: jnp.stack(xs), *all_states)

xs_full_trajectory = jnp.concatenate([init_state.obs[:3].reshape(1, -1), full_trajectory.obs, ])
rewards_full_trajectory = jnp.concatenate([init_state.reward.reshape(1, ), full_trajectory.reward])
# ts_full_trajectory = jnp.linspace(0, env.time_horizon, episode_length)
ts_full_trajectory = jnp.arange(0, xs_full_trajectory.shape[0]) * env.env.dt

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
xs = trajectory[0][:, :-1]
us = trajectory[1][:, :-1]
rewards = trajectory[2]
if time_as_part_of_state:
    times = trajectory[0][:, -1]
else:
    times = all_states.pipeline_state.time
times_for_actions = trajectory[1][:, -1]

total_time = env.time_horizon
# All times are the times when we ended the actions
all_ts = times
all_ts = jnp.concatenate([jnp.array([0.0]), all_ts])

all_xs = jnp.concatenate([state.obs[:-1].reshape(1, -1), xs])

state_dict = {0: r'cos($\theta$)',
              1: r'sin($\theta$)',
              2: r'$\omega$'}

if PLOT_TRUE_TRAJECTORIES:
    for i in range(3):
        axs[0].plot(ts_full_trajectory, xs_full_trajectory[:, i], label=state_dict[i])
    for h in all_ts[:-1]:
        axs[0].axvline(x=h, color='black', ls='--', alpha=0.4)
else:
    for i in range(3):
        axs[0].plot(all_ts, all_xs[:, i], label=state_dict[i])

axs[0].set_xlabel('Time', fontsize=LABEL_SIZE)
axs[0].set_ylabel('State', fontsize=LABEL_SIZE)

axs[1].step(all_ts, jnp.concatenate([us, us[-1].reshape(1, -1)]), where='post', label=r'$u$')
axs[1].set_xlabel('Time', fontsize=LABEL_SIZE)
axs[1].set_ylabel('Action', fontsize=LABEL_SIZE)

integrated_rewards = rewards / jnp.diff(all_ts) * 0.05

if PLOT_TRUE_TRAJECTORIES:
    axs[2].plot(ts_full_trajectory, rewards_full_trajectory, label='Rewards')
    for h in all_ts[:-1]:
        axs[2].axvline(x=h, color='black', ls='--', alpha=0.4)
else:
    axs[2].step(all_ts, jnp.concatenate([integrated_rewards, integrated_rewards[-1].reshape(1, )]),
                where='post', label='Rewards')
axs[2].set_xlabel('Time', fontsize=LABEL_SIZE)
axs[2].set_ylabel('Instance reward', fontsize=LABEL_SIZE)

axs[3].plot(jnp.diff(all_ts), label='Times for actions')
axs[3].set_xlabel('Action Steps', fontsize=LABEL_SIZE)
axs[3].set_ylabel('Time for action', fontsize=LABEL_SIZE)

# axs[4].plot(times_to_go, label='Time to go')
# axs[3].plot(times_for_actions, label='Time for actions NORMALIZED')
for ax in axs:
    ax.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
plt.savefig('pendulum_switch_cost.pdf')
plt.show()
print(f'Total reward: {jnp.sum(rewards_full_trajectory[:episode_len])}')
print(f'Total number of actions {len(us)}')

wandb.finish()
