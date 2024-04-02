import jax
import jax.numpy as jnp
from brax import base
from brax.envs.base import PipelineEnv, State, Env
from brax.io import mjcf
import mujoco
from pathlib import Path
from mujoco import mjx
import imageio


class Crazyflie2(PipelineEnv):

    def __init__(self, backend='mjx', **kwargs):
        # model_dir = Path('../../mujoco_menagerie/bitcraze_crazyflie_2')
        # model_xml = model_dir / "scene.xml"

        model_dir = Path('../assets/fetch')
        model_xml = model_dir / "pick_and_place.xml"

        assert backend == 'mjx'

        mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
        sys = mjcf.load_model(mj_model)
        super().__init__(sys, **kwargs)
        x = 10

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        q = self.sys.init_q
        qd = jnp.zeros(shape=(self.sys.qd_size(),))
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jnp.zeros(2)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = 1.0
        done = jnp.zeros(1)
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

    env = Crazyflie2()
    state = env.reset(rng=jax.random.PRNGKey(0))
    action = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(4,), minval=-1, maxval=1)
    state = env.step(state, action)

    # video_frames = env.render([state.pipeline_state] * 100, camera='track')
    #
    # new_filename = 'drone.mp4'
    #
    # with imageio.get_writer(new_filename, fps=int(1 / env.dt)) as writer:
    #     for frame in video_frames:
    #         writer.append_data(frame)
