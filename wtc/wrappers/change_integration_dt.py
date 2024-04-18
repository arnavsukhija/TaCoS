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
        next_step = next_step.replace(reward=next_step.reward / self.dt_divisor * self.action_repeat)
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
