import jax
import jax.numpy as jnp
from brax.envs.base import State, Env
from jax import vmap, jit
from jax.lax import cond
from jax.scipy.integrate import trapezoid
from jax.scipy.linalg import expm
from jaxtyping import Float, Array, Int

from wtc.utils.tolerance_reward import ToleranceReward

EPS = 1e-10


class LinearDynamicsBoundedSwitches(Env):
    def __init__(self,
                 a: Float[Array, 'dim_x dim_x'],
                 b: Float[Array, 'dim_x dim_u'],
                 x0: Float[Array, 'dim_x'],
                 time_horizon: Float[Array, "None"],
                 number_of_switches: Int[Array, "None"],
                 min_time_between_switches: Float[Array, "None"],
                 max_time_between_switches: Float[Array, "None"],
                 number_of_reward_evaluations: int,
                 discount_factor: float = 0.2,
                 q: Float[Array, 'dim_x dim_x'] | None = None,
                 r: Float[Array, 'dim_u dim_u'] | None = None,
                 reward_type: str = 'lqr'  # Options are lqr or dm-control
                 ):
        self.a = a
        self.b = b
        self.x0 = x0
        self.dim_x, self.dim_u = b.shape
        self.time_horizon = time_horizon
        self.number_of_switches = number_of_switches
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches
        self.number_of_reward_evaluations = number_of_reward_evaluations
        bound, value_at_margin, margin_factor = 0.1, 0.1, 10.0
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')
        self.discount_factor = discount_factor
        if q is None:
            q = jnp.eye(self.dim_x)
        if r is None:
            r = jnp.eye(self.dim_u)
        self.q, self.r = q, r

        match reward_type:
            case 'lqr':
                self.running_reward = self.running_reward_quadratic
            case 'dm-control':
                self.running_reward = self.running_reward_tolerance
            case _:
                raise ValueError(f'Reward type {reward_type} not recognized, possible values are lqr, dm-control')

    def reset(self, rng: jax.Array) -> State:
        augmented_state = jnp.concatenate([self.x0, self.time_horizon.reshape(1), self.number_of_switches.reshape(1)])
        reward, done = jnp.zeros(2)
        metrics = {}
        return State(None, augmented_state, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        obs, time_to_go, num_remaining_switches = state.obs[:-2], state.obs[-2], state.obs[-1]
        u, pseudo_time_for_action = action[:-1], action[-1]

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, time_to_go]
        t_lower = self.min_time_between_switches
        t_upper = jnp.minimum(time_to_go, self.max_time_between_switches)

        def true_fn_action_time(t_lower, t_upper, pseudo_time_for_action):
            return t_upper + EPS, jnp.array(1)

        def false_fn_action_time(t_lower, t_upper, pseudo_time_for_action):
            return ((t_upper - t_lower) / 2 * pseudo_time_for_action + (t_upper + t_lower) / 2).reshape(), jnp.array(0)

        time_for_action, done = cond(t_upper <= t_lower,
                                     true_fn_action_time, false_fn_action_time,
                                     t_lower, t_upper, pseudo_time_for_action)

        def last_action_true_fn(time_for_action, done):
            return time_to_go + EPS, jnp.array(1)

        def last_action_false_fn(time_for_action, done):
            return time_for_action, done

        time_for_action, done = cond(num_remaining_switches == 1,
                                     last_action_true_fn, last_action_false_fn,
                                     time_for_action, done)

        x_next = self._step(x=obs, u=u, t=time_for_action)

        # Here we compute the total reward on the time horizon
        ts = jnp.linspace(0, time_to_go, num=self.number_of_reward_evaluations)
        xs = vmap(self._step, in_axes=(None, None, 0))(obs, u, ts)
        discount_factor = jnp.exp(-self.discount_factor * ts)
        rewards = vmap(self.running_reward, in_axes=(0, None))(xs, u)
        reward = trapezoid(y=rewards * discount_factor, x=ts)

        # Done can come from running out of time, number of switches or since we overpassed the horizon
        # Prepare augmented obs
        next_time_to_go = (time_to_go - time_for_action).reshape(1)
        next_num_remaining_switches = (num_remaining_switches - 1).reshape(1)
        augmented_next_obs = jnp.concatenate([x_next, next_time_to_go, next_num_remaining_switches])

        next_done = 1 - (1 - state.done) * (1 - done)
        augmented_next_state = state.replace(obs=augmented_next_obs,
                                             reward=reward,
                                             done=next_done
                                             )
        return augmented_next_state

    def _step(self,
              x: Float[Array, "dim_x"],
              u: Float[Array, "dim_u"],
              t: Float[Array, "None"]
              ) -> Float[Array, "dim_x"]:
        e_a = expm(self.a * t)
        a_inv = jnp.linalg.inv(self.a)
        return e_a @ x + (e_a - jnp.eye(e_a.shape[0])) @ a_inv @ self.b @ u

    def running_reward_tolerance(self,
                                 x: Float[Array, "dim_x"],
                                 u: Float[Array, "dim_u"],
                                 ) -> Float[Array, "None"]:
        reward = self.tolerance_reward(jnp.sqrt(jnp.sum(x ** 2))) - 0.01 * jnp.sum(u ** 2)
        return reward

    def running_reward_quadratic(self,
                                 x: Float[Array, "dim_x"],
                                 u: Float[Array, "dim_u"],
                                 ) -> Float[Array, "None"]:
        lqr_cost = x.T @ self.q @ x + u.T @ self.r @ u
        return -lqr_cost

    @property
    def observation_size(self) -> int:
        return self.dim_x + 2

    @property
    def action_size(self) -> int:
        return self.dim_u + 1

    @property
    def backend(self) -> str:
        return 'generalized'


if __name__ == '__main__':
    from wtc.utils.create_system_matrix import create_marginally_stable_matrix
    import jax.random as jr
    import time

    key = jr.PRNGKey(42)
    x_dim, u_dim = 4, 4
    a = create_marginally_stable_matrix(x_dim, key=key)
    b = jnp.eye(u_dim)
    x0 = jr.uniform(shape=(x_dim,), key=key)

    env = LinearDynamicsBoundedSwitches(
        a=a,
        b=b,
        x0=x0,
        time_horizon=jnp.array(10.0),
        number_of_switches=jnp.array(20),
        min_time_between_switches=jnp.array(0.1),
        max_time_between_switches=jnp.array(5.0),
        number_of_reward_evaluations=10,
        reward_type='lqr'
    )

    state = env.reset(key)
    action = 0.1 * jnp.concatenate([jnp.ones(shape=(u_dim,)), jnp.array([0.1])])
    next_state = env.step(state, action=action)

    jitted_step = jit(env.step)

    for i in range(10):
        start_time = time.time()
        state = jitted_step(state, action)
        print(f'Reward: {state.reward}')
        print(time.time() - start_time)
