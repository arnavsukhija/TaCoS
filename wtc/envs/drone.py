from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from wtc.utils.tolerance_reward import ToleranceReward


class Crazyflie2(PipelineEnv):

    def __init__(self, backend='mjx', **kwargs):
        model_dir = Path('../../mujoco_menagerie/bitcraze_crazyflie_2')
        model_xml = model_dir / "scene.xml"
        assert backend == 'mjx'

        mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
        sys = mjcf.load_model(mj_model)
        super().__init__(sys, **kwargs)
        # We move one meter up and also in both x any coordinate
        self.target_state = jnp.array([1.0, 1.0, 1.0, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        bound, value_at_margin, margin_factor = 0.1, 0.1, 10.0
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        q = self.sys.init_q
        qd = jnp.zeros(shape=(self.sys.qd_size(),))
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jnp.zeros(2)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics)

    def reward(self, obs: chex.Array, action: chex.Array) -> float:
        """Computes the reward"""

        reward = self.tolerance_reward(jnp.sqrt(jnp.sum(obs ** 2))) - 0.01 * jnp.sum(action ** 2)
        return reward

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = self.reward(obs, action)
        done = jnp.zeros(1).reshape()
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 4

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jnp.concatenate([pipeline_state.qpos, pipeline_state.qvel])

    def backend(self) -> str:
        return "mjx"


if __name__ == '__main__':
    import imageio

    print('We started')
    env = Crazyflie2()
    state = env.reset(rng=jax.random.PRNGKey(0))
    print('We created step')
    action = jnp.array([0.35, 0.0, 0.0, 0.0])
    jitted_step = jax.jit(env.step)

    full_states = []
    pipeline_states = []

    for i in range(2000):
        state = jitted_step(state, action)
        full_states.append(state)
        pipeline_states.append(state.pipeline_state)

    video_frames = env.render(pipeline_states, camera='track')

    new_filename = 'drone.mp4'

    with imageio.get_writer(new_filename, fps=int(1 / env.dt)) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    import numpy as np

    np.set_printoptions(precision=2, suppress=True)

    obs = [state.obs for state in full_states]
    obs = jnp.stack(obs)

    import time

    jitted_reset = jax.jit(env.reset)
    for i in range(5):
        start = time.time()
        init_state = jitted_reset(rng=jax.random.PRNGKey(0))
        print("Time passed: ", time.time() - start)
