from typing import Any

import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper
import jax


class MinimalWrapper(mjx_env.MjxEnv):
    def __init__(self, env):
        self.env = env
        self._mjx_model = self.env.mjx_model
        self._mj_model = self.env.mj_model
        self._unwrapped = self.env.unwrapped
        self._xml_path = self.env.xml_path

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self.env.step(state, action)

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def unwrapped(self) -> Any:
        return self._unwrapped