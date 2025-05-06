from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper
import jax


class MinimalWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self.env.step(state, action)