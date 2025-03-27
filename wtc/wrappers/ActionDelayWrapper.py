import jax
import jax.numpy as jnp
from brax.envs.base import State, Env, Wrapper


class ActionDelayWrapper(Wrapper):
    """
    Brax wrapper that adds an action-delay to the base environment (a transformation of the underlying MDP to a new augmented MDP)
    """
    def __init__(self, env: Env, action_delay: float, ctrl_diff_weight: float = 0.01):
        super().__init__(env)
        self.ctrl_diff_weight = ctrl_diff_weight

        # Buffer parameters for action delay buffer
        self.dt = env.dt
        delay_steps = action_delay / self.dt  # calculates the number of delay steps
        self.buffer_size = int(jnp.ceil(delay_steps)) + 1
        if abs(action_delay % self.dt) < 1e-8:
            self.interp_weights = jnp.array([1.0, 0.0])
        else:
            weight_on_first = (action_delay % self.dt) / self.dt
            self.interp_weights = jnp.array([weight_on_first, 1.0 - weight_on_first])

    def reset(self, rng: jax.Array) -> State:
        "Uses the environments reset function and adds the action_delay buffer to the new state as part of info (easy maintenance), to maintain Markovian property"
        state = self.env.reset(rng)
        action_buffer = jnp.zeros((self.buffer_size, self.action_size))
        new_obs = {'observation': state.obs, 'action_buffer': action_buffer}
        return state.replace(obs=new_obs)

    def step(self, state: State, action: jax.Array) -> State:
        """We take a step with delayed action"""
        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        action_buffer = state.obs['action_buffer']
        # we reset the original structure of the state so that the base environment can process it easily
        obs = state.obs['observation']
        state = state.replace(obs=obs)
        delayed_action = jnp.sum(action_buffer[:2] * self.interp_weights[:, None], axis=0)
        next_state = self.env.step(state, delayed_action)
        # we derive the new action buffer and transform the state accordingly
        new_action_buffer = jnp.concatenate([state.action_buffer[1:], action[None]], axis=0)
        new_obs = {'observation': next_state.obs, 'action_buffer': new_action_buffer}
        control_penalty = -self.ctrl_diff_weight * jnp.sum((action - state.action_buffer[
            -1]) ** 2)  #compute control penalty based on the predicted action and the last action in buffer
        return next_state.replace(obs=new_obs, reward=next_state.reward + control_penalty)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size
