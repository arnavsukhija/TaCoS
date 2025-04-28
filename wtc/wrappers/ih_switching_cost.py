from abc import abstractmethod
from functools import partial
from typing import NamedTuple, Mapping, Optional, Callable, Tuple, Any

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import mujoco
from brax.envs.base import PipelineEnv, State, Env, base
from jax import jit
from jax.lax import while_loop, scan
from jaxtyping import Float, Array
from mujoco import mjx

EPS = 1e-10


class AugmentedPipelineState(NamedTuple):
    pipeline_state: base.State
    time: Float[Array, 'None']


class SwitchCost:
    @abstractmethod
    def __call__(self,
                 state: Float[Array, 'observation_dim'],
                 action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        pass


class ConstantSwitchCost(SwitchCost):

    def __init__(self, value: Float[Array, 'None']):
        self.value = value

    @partial(jit, static_argnums=(0,))
    def __call__(self,
                 state: Float[Array, 'observation_dim'],
                 action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return self.value


class IHSwitchCostWrapper(Env):
    def __init__(self,
                 env: PipelineEnv,
                 episode_steps: int, #number of steps for each reward integration (dt term in integration)
                 min_time_between_switches: float, # corresponds to tmin
                 max_time_between_switches: float | None = None, #corresponds to tmax
                 switch_cost: SwitchCost = ConstantSwitchCost(value=jnp.array(1.0)), #we use a default constant switch cost of 1.0
                 discounting: float = 0.99,
                 time_as_part_of_state: bool = False,
                 ismujoco_env: bool = False,
                 sim_dt: float = 1,
                 env_randomization_fn: Optional[
                     Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
                 ] = None,
                 ):
        self.env = env
        self.episode_steps = episode_steps
        self.num_integrator_steps = episode_steps * env.dt / sim_dt
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches
        assert min_time_between_switches >= self.env.dt, \
            'Min time between switches must be at least of the integration time dt' #otherwise the integration term makes no sense at all
        self.time_horizon = self.env.dt * episode_steps  #this corresponds to the T from the paper, should be
        if max_time_between_switches is None:
            max_time_between_switches = self.time_horizon
        self.max_time_between_switches = max_time_between_switches
        self.discounting = discounting
        self.time_as_part_of_state = time_as_part_of_state #this includes the state definition, for interaction cost time is part of the state
        self.jitted_step_fn = jit(self.env.step)
        self.ismujoco_env = ismujoco_env
        self.env_randomization_fn = env_randomization_fn


    def randomization_fn(self, model: mjx.Model, rng:jax.Array):
        return self.env_randomization_fn(model, rng)
    def _add_time_to_obs(self, state: State, time: jax.Array) -> State:
        # we handle the case where it is a state from a Mujoco Env
        if self.ismujoco_env:
            state_obs_from_dic = state.obs['state']
            privileged_state = state.obs['privileged_state']
            augmented_state_obs_from_dic = jnp.concatenate([state_obs_from_dic, time.reshape(1)])
            augmented_privileged_state = jnp.concatenate([privileged_state[:state_obs_from_dic.size], time.reshape(1),
                                                          privileged_state[state_obs_from_dic.size:]])
            augmented_obs = {
                'state': augmented_state_obs_from_dic,
                'privileged_state': augmented_privileged_state
            }
        else:
            augmented_obs = jnp.concatenate([state.obs, time.reshape(1)])
        return augmented_obs

    def reset(self, rng: jax.Array) -> State:
        """
        The augmented state is represented by concatenated vector: #also includes reward (implicit in the state)
         (state, time-to-go)
        """
        state = self.env.reset(rng)
        time = jnp.array(0)
        if self.time_as_part_of_state:
            # we check whether the state observation is a jax.Array or a mapping, and extract the obs accordingly for the concatenation
            augmented_obs = self._add_time_to_obs(state, time)
            augmented_state = state.replace(obs=augmented_obs)
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=state.pipeline_state,
                                                              time=time)
            augmented_state = state.replace(pipeline_state=augmented_pipeline_state)
        return augmented_state

    def compute_time(self,
                     pseudo_time: chex.Array,
                     t_lower: chex.Array, # pass this as steps now, not time
                     t_upper: chex.Array, # pass this as steps now, not time
                     ) -> chex.Array:
        time_for_action = ((t_upper - t_lower) / 2 * pseudo_time + (t_upper + t_lower) / 2) #pseudo time for action is between [-1,1], we map it to tmin, tmax
        return jnp.floor(time_for_action)

    def _get_time_and_obs(self, state: State):
        if self.ismujoco_env:
            obs, time, = state.obs['state'][:-1], state.obs['state'][-1]
            obs_size = obs.size
            privileged_state = jnp.concatenate([state.obs['privileged_state'][:obs_size], state.obs['privileged_state'][obs_size+1:]]) # only applicable if value state is different than policy state
        else:
            obs, time, privileged_state = state.obs[:-1], state.obs[-1], []
        return obs, time, privileged_state

    def step(self, state: State, action: jax.Array) -> State:
        u, pseudo_time_for_action = action[:-1], action[-1]
        if self.time_as_part_of_state:
            obs, time, privileged_state_obs = self._get_time_and_obs(state)
        else:
            env_pipeline_state = state.pipeline_state.pipeline_state
            time = state.pipeline_state.time

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, self.max_time_between_switches] (corresponds to number of steps now)
        steps_to_apply = self.compute_time(pseudo_time=pseudo_time_for_action,
                                            t_lower=self.min_time_between_switches,
                                            t_upper=self.max_time_between_switches,
                                            )

        done = steps_to_apply >= self.episode_steps - time
        # Calculate how many steps we need to take with action
        num_steps = jnp.minimum(steps_to_apply, self.episode_steps - time) #calculate how often we apply this action based on the environment dt

        # Integrate dynamics forward for the num_steps
        if self.ismujoco_env:
            old_obs = {
                'state': obs,
                'privileged_state': privileged_state_obs
            }
        else:
            old_obs = obs
        if self.time_as_part_of_state:
            state = state.replace(obs=old_obs)
        else:
            state = state.replace(pipeline_state=env_pipeline_state)

        def body_integration_step(val):
            s, r, index = val
            next_state = self.env.step(s, u)
            next_reward = r + (self.discounting ** index) * (1 - next_state.done) * next_state.reward
            return next_state, next_reward, index + 1

        def cond_integration_step(val):
            s, r, index = val
            # We continue if index is smaller that num_steps ant we are not done
            return jnp.bitwise_and(index < num_steps, jnp.bitwise_not(s.done.astype(bool)))

        init_val = (state, jnp.array(0.0), jnp.array(0))
        final_val = while_loop(cond_integration_step, body_integration_step, init_val)
        next_state, total_reward, index = final_val
        next_done = 1 - (1 - next_state.done) * (1 - done)

        # Add switch cost to the total reward
        total_reward = total_reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs (how many steps we actually took)
        next_time = (time + index)
        if self.time_as_part_of_state:
            augmented_next_obs = self._add_time_to_obs(next_state, next_time)
            augmented_next_state = next_state.replace(obs=augmented_next_obs,
                                                      reward=total_reward,
                                                      done=next_done)
            return augmented_next_state
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=next_state.pipeline_state,
                                                              time=next_time)
            augmented_next_state = next_state.replace(reward=total_reward,
                                                      done=next_done,
                                                      pipeline_state=augmented_pipeline_state)
            return augmented_next_state

    # TODO: This function is now not jittable (it's on purpose)
    def simulation_step(self, state: State, action: jax.Array) -> (State, State):
        u, pseudo_time_for_action = action[:-1], action[-1]
        if self.time_as_part_of_state:
            obs, time, privileged_state_obs = self._get_time_and_obs(state)
        else:
            env_pipeline_state = state.pipeline_state.pipeline_state
            time = state.pipeline_state.time

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, time_to_go] (now number of steps)
        steps_to_apply = self.compute_time(pseudo_time=pseudo_time_for_action,
                                            t_lower=self.min_time_between_switches,
                                            t_upper=self.max_time_between_switches)
        done = steps_to_apply >= self.episode_steps - time

        # Calculate how many steps we need to take with action
        num_steps = jnp.minimum(steps_to_apply, self.episode_steps - time)

        # Integrate dynamics forward for the num_steps
        if self.ismujoco_env:
            old_obs = {
                'state': obs,
                'privileged_state': privileged_state_obs
            }
        else:
            old_obs = obs
        if self.time_as_part_of_state:
            state = state.replace(obs=old_obs)
        else:
            state = state.replace(pipeline_state=env_pipeline_state)

        # Execute the action for the predicted number of integration steps
        step_index = 0
        cur_state = state
        all_states = []
        while step_index < num_steps and not cur_state.done:
            cur_state = self.jitted_step_fn(cur_state, u)
            all_states.append(cur_state)
            step_index += 1

        next_state = cur_state
        if len(all_states) == 0:
            all_states = [state]
        inner_part = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_states)
        total_reward = jnp.sum(inner_part.reward)
        next_done = 1 - (1 - next_state.done) * (1 - done)

        # Add switch cost to the total reward
        total_reward = total_reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs
        next_time = (time + step_index)
        if self.time_as_part_of_state:
            augmented_next_obs = self._add_time_to_obs(next_state, next_time)
            augmented_next_state = next_state.replace(obs=augmented_next_obs,
                                                      reward=total_reward,
                                                      done=next_done)
            return augmented_next_state, all_states
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=next_state.pipeline_state,
                                                              time=next_time.reshape())
            augmented_next_state = next_state.replace(reward=total_reward,
                                                      done=next_done,
                                                      pipeline_state=augmented_pipeline_state)
            return augmented_next_state, all_states

    @property
    def observation_size(self):
        # +1 for time-to-go ant +1 for num remaining switches
        if self.time_as_part_of_state:
            if self.ismujoco_env:
                obs_size = {
                    k: (v[0]+1, ) for k, v in self.env.observation_size.items()
                }
            else:
                obs_size = self.env.observation_size + 1
            return obs_size
        else:
            return self.env.observation_size
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

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self.env.mjx_model

    @property
    def xml_path(self) -> str:
        return self.env.xml_path


if __name__ == '__main__':
    from brax import envs
    import jax.random as jr
    from jax import jit

    env_name = 'inverted_pendulum'
    backend = 'generalized'

    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    env = IHSwitchCostWrapper(env,
                              episode_steps=1000,
                              min_time_between_switches=env.dt,
                              # max_time_between_switches=10 * env.dt,
                              switch_cost=ConstantSwitchCost(value=jnp.array(1.0)),
                              discounting=1.0)

    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    state = env.reset(subkey)

    wrapper = True

    u = jnp.array([0.1])
    time = jnp.array([0.0])
    augmented_action = jnp.concatenate([u, time]) if wrapper else u

    state, rest = env.simulation_step(state, augmented_action)
    state = env.step(state, augmented_action)
    # jitted_step = jit(env.step)

    # import time
    #
    # for i in range(10):
    #     start_time = time.time()
    #     state = jitted_step(state, augmented_action)
    #     print(f'elapsed_time: {time.time() - start_time} sec')
