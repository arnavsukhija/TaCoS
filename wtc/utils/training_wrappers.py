from typing import Callable, Optional, Tuple
import jax
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as brax_training
from jax import numpy as jp

class DomainRandomizationVmapBase(Wrapper):
    """Base class for domain randomization wrappers."""

    def __init__(self, env, randomization_fn, *, augment_state=True):
        super().__init__(env)
        self.augment_state = augment_state
        (
            self._randomized_models,
            self._in_axes,
            self.domain_parameters,
        ) = self._init_randomization(randomization_fn)
        dummy = self.env.reset(jax.random.PRNGKey(0))
        self.strip_privileged_state = isinstance(dummy.obs, jax.Array)

    def _init_randomization(self, randomization_fn):
        """To be implemented by subclasses to handle model-specific randomization."""
        raise NotImplementedError

    def _env_fn(self, model):
        """To be implemented by subclasses to return an environment with the given model."""
        raise NotImplementedError

    def reset(self, rng: jax.Array):
        def reset_fn(model, rng):
            env = self._env_fn(model)
            return env.reset(rng)

        state = jax.vmap(reset_fn, in_axes=[self._in_axes, 0])(
            self._randomized_models, rng
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def step(self, state, action: jax.Array):
        def step_fn(model, s, a):
            env = self._env_fn(model)
            return env.step(s, a)

        if self.augment_state and self.strip_privileged_state:
            state = state.replace(obs=state.obs["state"])

        state = jax.vmap(step_fn, in_axes=[self._in_axes, 0, 0])(
            self._randomized_models, state, action
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def _add_privileged_state(self, state):
        """Adds privileged state to the observation if augmentation is enabled."""
        if isinstance(state.obs, jax.Array):
            state = state.replace(
                obs={
                    "state": state.obs,
                    "privileged_state": jp.concatenate(
                        [state.obs, self.domain_parameters], -1
                    ),
                }
            )
        else:
            state = state.replace(
                obs={
                    "state": state.obs["state"],
                    "privileged_state": jp.concatenate(
                        [state.obs["privileged_state"], self.domain_parameters], -1
                    ),
                }
            )
        return state

    @property
    def observation_size(self):
        """Compute observation size based on the augmentation setting."""
        if not self.augment_state:
            return self.env.observation_size

        if isinstance(self.env.observation_size, int):
            return {
                "state": (self.env.observation_size,),
                "privileged_state": (
                    self.env.observation_size + self.domain_parameters.shape[1],
                ),
            }
        else:
            return {
                "state": (self.env.observation_size["state"],),
                "privileged_state": (
                    self.env.observation_size["privileged_state"]
                    + self.domain_parameters.shape[1],
                ),
            }


class DomainRandomizationVmapWrapper(DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        return randomization_fn(self.sys)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped._dynamics_params = model
        return env


class CostEpisodeWrapper(brax_training.EpisodeWrapper):
    """Maintains episode step count and sets done at episode end."""

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            maybe_cost = nstate.info.get("cost", None)
            return nstate, (nstate.reward, maybe_cost)

        state, (rewards, maybe_costs) = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        if maybe_costs is not None:
            state.info["cost"] = jp.sum(maybe_costs, axis=0)
        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps
        return state.replace(done=done)


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System, jax.Array]]
    ] = None,
    augment_state: bool = True,
) -> Wrapper:
    """Common wrapper pattern for all training agents, including switch cost wrapper

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(
            env, randomization_fn, augment_state=augment_state
        )
    env = brax_training.AutoResetWrapper(env)
    return env