import argparse
import datetime
import os
import pickle
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax import envs
from jax.nn import swish
from mbpo.optimizers.policy_optimizers.ppo.ppo_brax_env import PPO

ENTITY = 'trevenl'


def experiment(env_name: str = 'ant',
               backend: str = 'generalized',
               project_name: str = 'GPUSpeedTest',
               num_timesteps: int = 50_000_000,
               episode_length: int = 200,
               action_repeat: int = 1,
               num_envs: int = 4096,
               num_eval_envs: int = 128,
               learning_rate: float = 3e-4,
               entropy_cost: float = 1e-2,
               discounting: float = 0.97,
               seed: int = 1,
               unroll_length: int = 5,
               batch_size: int = 2048,
               num_minibatches: int = 32,
               num_updates_per_batch: int = 4,
               num_evals: int = 10,
               normalize_observations: bool = True,
               reward_scaling: float = 10.0,
               clipping_epsilon: float = 0.3,
               gae_lambda: float = 0.95,
               deterministic_eval: bool = False,
               normalize_advantage: bool = True,
               ):
    assert env_name in ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
                        'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d', 'drone', 'greenhouse',
                        'swimmer']
    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    config = dict(env_name=env_name,
                  num_timesteps=num_timesteps,
                  episode_length=episode_length,
                  action_repeat=action_repeat,
                  num_envs=num_envs,
                  num_eval_envs=num_eval_envs,
                  lr=learning_rate,
                  wd=0.,
                  entropy_cost=entropy_cost,
                  unroll_length=unroll_length,
                  discounting=discounting,
                  batch_size=batch_size,
                  num_minibatches=num_minibatches,
                  num_updates_per_batch=num_updates_per_batch,
                  num_evals=num_evals,
                  normalize_observations=normalize_observations,
                  reward_scaling=reward_scaling,
                  clipping_epsilon=clipping_epsilon,
                  gae_lambda=gae_lambda,
                  policy_hidden_layer_sizes=(32,) * 4,
                  policy_activation=swish,
                  critic_hidden_layer_sizes=(256,) * 5,
                  critic_activation=swish,
                  deterministic_eval=deterministic_eval,
                  normalize_advantage=normalize_advantage,
                  )

    wandb.init(
        project=project_name,
        dir='/cluster/scratch/' + ENTITY,
        config=config,
    )

    optimizer = PPO(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        action_repeat=action_repeat,
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        lr=learning_rate,
        wd=0.,
        entropy_cost=entropy_cost,
        unroll_length=unroll_length,
        discounting=discounting,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        num_evals=num_evals,
        normalize_observations=normalize_observations,
        reward_scaling=reward_scaling,
        clipping_epsilon=clipping_epsilon,
        gae_lambda=gae_lambda,
        policy_hidden_layer_sizes=(32,) * 4,
        policy_activation=swish,
        critic_hidden_layer_sizes=(256,) * 5,
        critic_activation=swish,
        deterministic_eval=deterministic_eval,
        normalize_advantage=normalize_advantage,
        wandb_logging=True,
    )

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        plt.show()

    print('Before inference')
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(seed), progress_fn=progress)
    print('After inference')

    # Now we plot the evolution
    pseudo_policy = optimizer.make_policy(policy_params, deterministic=True)

    @jax.jit
    def policy(obs):
        return pseudo_policy(obs, key_sample=jr.PRNGKey(0))

    ########################## Evaluation ##########################
    ################################################################

    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    state = env.reset(rng=jr.PRNGKey(0))
    step_fn = jax.jit(env.step)

    trajectory = []
    total_steps = 0
    while (not state.done) and (total_steps < episode_length):
        action = policy(state.obs)[0]
        for _ in range(action_repeat):
            state = step_fn(state, action)
            total_steps += 1
            trajectory.append(state)

    trajectory = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trajectory)

    wandb.log({'results/total_reward': jnp.sum(trajectory.reward),
               'results/num_actions': len(trajectory.reward)})

    # We save full_trajectory to wandb
    # Save trajectory rather than rendered video
    directory = os.path.join(wandb.run.dir, 'results')
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, 'trajectory_1.pkl')
    with open(model_path, 'wb') as handle:
        pickle.dump(trajectory, handle)
    wandb.save(model_path, wandb.run.dir)
    wandb.finish()


def main(args):
    experiment(env_name=args.env_name,
               backend=args.backend,
               project_name=args.project_name,
               num_timesteps=args.num_timesteps,
               episode_length=args.episode_length,
               action_repeat=args.action_repeat,
               num_envs=args.num_envs,
               num_eval_envs=args.num_eval_envs,
               learning_rate=args.learning_rate,
               entropy_cost=args.entropy_cost,
               discounting=args.discounting,
               seed=args.seed,
               unroll_length=args.unroll_length,
               batch_size=args.batch_size,
               num_minibatches=args.num_minibatches,
               num_updates_per_batch=args.num_updates_per_batch,
               num_evals=args.num_evals,
               normalize_observations=bool(args.normalize_observations),
               reward_scaling=args.reward_scaling,
               clipping_epsilon=args.clipping_epsilon,
               gae_lambda=args.gae_lambda,
               deterministic_eval=bool(args.deterministic_eval),
               normalize_advantage=bool(args.normalize_advantage),
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='swimmer')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='GPUSpeedTest')
    parser.add_argument('--num_timesteps', type=int, default=100_000)
    parser.add_argument('--episode_length', type=int, default=200)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--num_eval_envs', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--num_evals', type=int, default=10)
    parser.add_argument('--normalize_observations', type=int, default=1)
    parser.add_argument('--reward_scaling', type=float, default=10.0)
    parser.add_argument('--clipping_epsilon', type=float, default=0.3)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--deterministic_eval', type=int, default=1)
    parser.add_argument('--normalize_advantage', type=int, default=1)

    args = parser.parse_args()
    main(args)
