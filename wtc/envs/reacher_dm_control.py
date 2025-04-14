import jax
from brax.envs import reacher, State
from wtc.utils.tolerance_reward import ToleranceReward
import jax.numpy as jnp
import jax.random as jr


class ReacherDMControl(reacher.Reacher):
    def __init__(self, *args, seed=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        bound = 0.01
        value_at_margin = 0.2
        margin_factor = 10.0
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')

    def reset(self, rng: jax.Array) -> State:
        """
        We always reset to the same position
        """
        return super().reset(jr.PRNGKey(self.seed))

    def reward(self, obs, action):
        reward_dist = self.tolerance_reward(jnp.sqrt(jnp.sum(obs[-3:] ** 2)))
        reward_ctrl = -jnp.square(action).sum()
        reward = reward_dist + 0.01 * reward_ctrl
        return reward

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        # vector from tip to target is last 3 entries of obs vector
        reward_dist = self.tolerance_reward(jnp.sqrt(jnp.sum(obs[-3:] ** 2)))
        reward_ctrl = -jnp.square(action).sum()
        reward = reward_dist + 0.1 * reward_ctrl

        state.metrics.update(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)


if __name__ == '__main__':
    env = ReacherDMControl(backend='generalized')
    print(env.dt)
