from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from jaxtyping import Float, Array
from wtc.utils.tolerance_reward import ToleranceReward


class MountainCarEnv(Env):
    def __init__(self, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = jnp.array(
            [self.min_position, -self.max_speed], dtype=jnp.float32
        )
        self.high_state = jnp.array(
            [self.max_position, self.max_speed], dtype=jnp.float32
        )

        bound, value_at_margin, margin_factor = 0.1, 0.1, 10.0
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')

    def _reset(self, key: jax.random.PRNGKey):
        return jnp.array([jax.random.uniform(key=key, minval=-0.6, maxval=-0.4), 0])

    def reset(self,
              rng: jax.Array) -> State:
        return State(pipeline_state=None,
                     obs=self._reset(key=rng),
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0), )

    def ode(self, state, action):
        position = state[0]
        velocity = state[1]
        force = jnp.minimum(jnp.maximum(action[0], self.min_action), self.max_action)
        velocity = jnp.minimum(jnp.maximum(velocity, -self.max_speed), self.max_speed)
        return jnp.array([velocity, force * self.power - 0.0025 * jnp.cos(3 * position)])

    def reward(self, obs, action):
        position = obs[0]
        velocity = obs[1]
        distance_to_target = (position - self.goal_position) ** 2 + (velocity - self.goal_velocity) ** 2
        return self.tolerance_reward(jnp.sqrt(distance_to_target)) - 0.01 * jnp.sum(action ** 2)

    def step(self,
             state: State,
             action: jax.Array) -> State:
        obs = state.obs
        obs_dot = self.ode(obs, action)
        print(obs_dot)
        next_obs = obs + self.dt * obs_dot
        # print(obs, next_obs)
        position = next_obs[0]
        velocity = obs[1]
        position = jnp.minimum(jnp.maximum(position, self.min_position), self.max_position)
        next_obs = jnp.array([position, velocity])
        next_reward = self.reward(next_obs, action) * self.dt
        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    @property
    def dt(self):
        return 1.0

    @property
    def observation_size(self) -> int:
        return 2

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'


if __name__ == '__main__':
    env = MountainCarEnv()

    state = env.reset(rng=jr.PRNGKey(0))
    action = jnp.array([1.0])
    for i in range(100):
        state = env.step(state, action)
        print(state.obs)
