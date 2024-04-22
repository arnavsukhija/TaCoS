import jax
from brax.envs.base import PipelineEnv, State, Env


class ChangeIntegrationStep(Env):
    def __init__(self,
                 env: PipelineEnv,
                 dt_divisor: float = 1.0,
                 action_repeat: int = 1):
        self.dt_divisor = dt_divisor
        self.action_repeat = action_repeat
        self.env = env
        self.base_dt = self.env.sys.dt
        self.env._n_frames = env._n_frames * action_repeat
        self.env.sys = self.env.sys.replace(dt=self.base_dt / self.dt_divisor)

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        next_step = self.env.step(state, action)
        next_step = next_step.replace(reward=(next_step.reward / self.dt_divisor) * self.action_repeat)
        return next_step

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def backend(self) -> str:
        return self.env.backend

    @property
    def dt(self):
        return self.env.dt


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from brax import envs

    env_name = 'reacher'
    backend = 'generalized'

    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d', 'drone', 'greenhouse']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)
    print(env.dt)
    env = ChangeIntegrationStep(env=env,
                                action_repeat=5)
    print(env.dt)
    print(env.dt * 200)