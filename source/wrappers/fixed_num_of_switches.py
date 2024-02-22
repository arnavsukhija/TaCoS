from functools import partial

import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State, Env
from jax.lax import cond, while_loop

EPS = 1e-10


class FixedNumOfSwitchesWrapper(Env):
    def __init__(self,
                 env: PipelineEnv,
                 num_integrator_steps: int,
                 num_switches: int,
                 min_time_between_switches: float,
                 max_time_between_switches: float | None = None,
                 discounting: float = 0.99
                 ):
        self.env = env
        self.num_integrator_steps = num_integrator_steps
        self.num_switches = num_switches
        self.min_time_between_switches = min_time_between_switches
        assert min_time_between_switches >= self.env.dt, \
            'Min time between switches must be at least of the integration time dt'
        self.time_horizon = self.env.dt * self.num_integrator_steps
        if max_time_between_switches is None:
            max_time_between_switches = self.time_horizon
        self.max_time_between_switches = max_time_between_switches
        self.discounting = discounting

    def reset(self, rng: jax.Array) -> State:
        """
        The augmented state is represented by concatenated vector:
         (state, time-to-go, num_remaining_switches)
        """
        state = self.env.reset(rng)
        x0 = state.obs
        time_to_go = self.time_horizon
        num_remaining_switches = jnp.array(self.num_switches)
        augmented_obs = jnp.concatenate([x0, time_to_go.reshape(1), num_remaining_switches.reshape(1)])
        augmented_state = state.replace(obs=augmented_obs)
        # augmented_state = State(pipeline_state=state.pipeline_state,
        #                         obs=augmented_obs,
        #                         reward=state.reward,
        #                         done=state.done,
        #                         metrics=state.metrics,
        #                         info=state.info)
        return augmented_state

    # @partial(jax.jit, static_argnums=0)
    def step(self, state: State, action: jax.Array) -> State:
        obs, time_to_go, num_remaining_switches = state.obs[:-2], state.obs[-2], state.obs[-1]
        u, pseudo_time_for_action = action[:-1], action[-1]

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, time_to_go]
        t_lower = self.min_time_between_switches
        t_upper = jnp.minimum(time_to_go, self.max_time_between_switches)

        def true_fn_action_time(t_lower, t_upper, pseudo_time_for_action):
            return t_upper + EPS, True

        def false_fn_action_time(t_lower, t_upper, pseudo_time_for_action):
            return ((t_upper - t_lower) / 2 * pseudo_time_for_action + (t_upper + t_lower) / 2).reshape(), False

        time_for_action, done = cond(jnp.bitwise_or(t_upper <= t_lower, num_remaining_switches == 1),
                                     true_fn_action_time, false_fn_action_time,
                                     t_lower, t_upper, pseudo_time_for_action)

        # Calculate how many steps we need to take with action
        elapsed_time = self.time_horizon - time_to_go
        steps_passed = (elapsed_time // self.env.dt).astype(int)
        next_elapsed_time = elapsed_time + time_for_action
        next_steps_passed = (next_elapsed_time // self.env.dt).astype(int)
        num_steps = next_steps_passed - steps_passed
        # assert num_steps >= 1

        # Integrate dynamics forward for the num_steps
        state = state.replace(obs=obs, )

        def body_integration_step(val):
            s, r, index = val
            next_state = self.env.step(s, u)
            next_reward = r + (self.discounting ** index) * (1 - next_state.done) * next_state.reward
            return next_state, next_reward, index + 1

        def cond_integration_step(val):
            s, r, index = val
            # We continue if index is smaller that num_steps and we are not done
            return jnp.bitwise_and(index < num_steps, jnp.bitwise_not(s.done.astype(jnp.bool)))

        init_val = (state, jnp.array(0.0), jnp.array(0))
        final_val = while_loop(cond_integration_step, body_integration_step, init_val)

        next_state, total_reward, _ = final_val

        # Prepare augmented obs
        next_time_to_go = (time_to_go - time_for_action).reshape(1)
        next_num_remaining_switches = (num_remaining_switches - 1).reshape(1)
        augmented_next_obs = jnp.concatenate([next_state.obs, next_time_to_go, next_num_remaining_switches])

        augmented_next_state = next_state.replace(obs=augmented_next_obs, reward=total_reward)
        return augmented_next_state

    @property
    def observation_size(self) -> int:
        # +1 for time-to-go and +1 for num remaining switches
        return self.env.observation_size + 2

    @property
    def action_size(self) -> int:
        # +1 for time that we apply action for
        return self.env.action_size + 1

    @property
    def backend(self) -> str:
        return self.env.backend


if __name__ == '__main__':
    from brax import envs
    import jax.random as jr
    from jax import jit

    env_name = 'inverted_pendulum'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    env = FixedNumOfSwitchesWrapper(env,
                                    num_integrator_steps=1000,
                                    num_switches=30,
                                    min_time_between_switches=env.dt,
                                    max_time_between_switches=10 * env.dt,
                                    discounting=1.0)

    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    state = env.reset(subkey)

    wrapper = True

    u = jnp.array([0.1])
    time = jnp.array([0.0])
    augmented_action = jnp.concatenate([u, time]) if wrapper else u

    state = env.step(state, augmented_action)
    jitted_step = jit(env.step)

    import time

    for i in range(10):
        start_time = time.time()
        state = jitted_step(state, augmented_action)
        print(f'elapsed_time: {time.time() - start_time} sec')
