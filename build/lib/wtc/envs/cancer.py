from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from jaxtyping import Float, Array


class CancerEnv(Env):
    def __init__(self, reward_source: str = 'gym'):
        # self.rho = 7e-5 + 7.23e-3 * 4 # Good for inducing death
        self.e_noise = 0
        self.rho = 7e-5 + 7.23e-3 * 2
        self.K = self.calc_volume(30.0)
        self.beta_c = 0.028
        self.alpha_r = 0.0398
        self.beta_r = self.alpha_r / 10.0
        self.action_max = 2.0
        # self.nominal_chemo_drug = 5
        self.max_chemo_drug = 5.0

        # self.nominal_radio = 2.0
        self.max_radio = 2.0
        self.chemo_cost_scale = 0.1
        self.time_multiplier = 5.0
        self.v_death_thres = self.calc_volume(13.0)
        self.ac_rew_const = 0.001

    def calc_volume(self, diameter: Float[Array, 'None']):
        return 4.0 / 3.0 * jnp.pi * (diameter / 2.0) ** 3.0

    def calc_diameter(self, volume: Float[Array, 'None']):
        return ((volume / (4.0 / 3.0 * jnp.pi)) ** (1.0 / 3.0)) * 2.0

    def reset(self,
              rng: jax.Array) -> State:
        return State(pipeline_state=None,
                     obs=self._reset(key=rng),
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0), )

    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return self._reward(x, u)

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        obs = state.obs
        obs_dot = self.ode(obs, action)
        next_obs = obs + self.dt * obs_dot
        next_reward = self.reward(obs, action) * self.dt

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    def ode(self,
            x: Float[Array, 'observation_dim'],
            u: Float[Array, 'action_dim']) -> Float[Array, 'observation_dim']:
        """Input
            state  [N,n]
            action [N,m]
        Main simulation
        state::
        v = cancer_volume
        c = chemo_concentration

        action::
        ca = chemo_action (dosage)
        ra = radio_action (dosage)

        e = noise
        """
        e = self.e_noise
        rho = self.rho
        K = self.K
        beta_c = self.beta_c
        alpha_r = self.alpha_r
        beta_r = self.beta_r

        v, c = x[..., 0], x[..., 1]
        v = jnp.where(v <= 0, 0, v)

        # Here we scale action to [-self.action_max, self.action_max]
        u = 2 * u
        ca_unshifted, ra_unshifted = u[..., 0], u[..., 1]
        ca = (ca_unshifted / 2.0) * self.max_chemo_drug
        ra = (ra_unshifted / 2.0) * self.max_radio

        ca = jnp.where(ca <= 0, 0, ca)
        # ca[ca <= 0] = 0
        ra = jnp.where(ra <= 0, 0, ra)
        # ra[ra <= 0] = 0

        dc_dt = -c / 2 + ca
        dv_dt = (rho * jnp.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * jnp.square(ra)) + e) * v
        # dv_dt[v == 0] = 0
        dv_dt = jnp.where(v == 0, 0, dv_dt)
        dv_dt = jnp.nan_to_num(dv_dt, posinf=0, neginf=0)
        return jnp.stack([dv_dt * self.time_multiplier, dc_dt * self.time_multiplier], axis=-1)

    def diff_obs_reward_(self,
                         x: Float[Array, 'observation_dim'], ) -> Float[Array, 'None']:
        v, c = x[..., 0], x[..., 1]
        # v[v <= 0] = 0
        v = jnp.where(v <= 0, 0, v)

        state_reward = -jnp.square(self.calc_diameter(v))
        state_reward -= self.chemo_cost_scale * jnp.square(c)
        # state_reward -= (v>=self.v_death_thres).float() * 10000
        return state_reward

    def diff_ac_reward_(self,
                        action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return -self.ac_rew_const * jnp.sum(jnp.square((action / self.action_max)), -1)

    def _reward(self,
                x: Float[Array, 'observation_dim'],
                u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return self.diff_obs_reward_(x) + self.diff_ac_reward_(u)

    def _reset(self,
               key: chex.PRNGKey = jr.PRNGKey(0)) -> Float[Array, 'observation_dim']:
        state = jr.uniform(key=key,
                           minval=self.calc_volume(13.0) * 0.98,
                           maxval=self.calc_volume(13.0) * 0.99,
                           shape=(2,)).astype("float32")
        state = state.at[1].set(0.0) # Zero drug concentration starting
        return state

    @property
    def dt(self):
        return 0.1

    @property
    def observation_size(self) -> int:
        return 2

    @property
    def action_size(self) -> int:
        return 2

    def backend(self) -> str:
        return 'positional'


if __name__ == '__main__':
    env = CancerEnv()

    state = env.reset(rng=jr.PRNGKey(0))
    action = jnp.array([0.3, -0.2])
    for i in range(10):
        state = env.step(state, action)
