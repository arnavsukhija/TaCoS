import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'PPO_Apr_15_11_20'

ant_configs = {
    'env_name': ['ant'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [50_000_000, ],
    'episode_length': [200],
    'action_repeat': [1, ],
    'num_envs': [4096],
    'num_eval_envs': [128],
    'learning_rate': [3e-4],
    'entropy_cost': [1e-2],
    'discounting': [0.97],
    'seed': list(range(3)),
    'unroll_length': [5, ],
    'batch_size': [2048, ],
    'num_minibatches': [32, ],
    'num_updates_per_batch': [4, ],
    'num_evals': [10],
    'normalize_observations': [1, ],
    'reward_scaling': [10.0],
    'clipping_epsilon': [0.3],
    'gae_lambda': [0.95],
    'deterministic_eval': [1, ],
    'normalize_advantage': [1, ],
}

humanoid_configs = {
    'env_name': ['humanoid'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [50_000_000, ],
    'episode_length': [200],
    'action_repeat': [1, ],
    'num_envs': [2048],
    'num_eval_envs': [128],
    'learning_rate': [3e-4],
    'entropy_cost': [1e-3],
    'discounting': [0.97],
    'seed': list(range(3)),
    'unroll_length': [10, ],
    'batch_size': [1024, ],
    'num_minibatches': [32, ],
    'num_updates_per_batch': [8, ],
    'num_evals': [10],
    'normalize_observations': [1, ],
    'reward_scaling': [0.1],
    'clipping_epsilon': [0.3],
    'gae_lambda': [0.95],
    'deterministic_eval': [1, ],
    'normalize_advantage': [1, ],
}

halfcheetah_configs = {
    'env_name': ['halfcheetah'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [50_000_000, ],
    'episode_length': [200],
    'action_repeat': [1, ],
    'num_envs': [2048],
    'num_eval_envs': [128],
    'learning_rate': [3e-4],
    'entropy_cost': [0.001],
    'discounting': [0.95],
    'seed': list(range(3)),
    'unroll_length': [20, ],
    'batch_size': [512, ],
    'num_minibatches': [32, ],
    'num_updates_per_batch': [8, ],
    'num_evals': [20],
    'normalize_observations': [1, ],
    'reward_scaling': [1.0],
    'clipping_epsilon': [0.3],
    'gae_lambda': [0.95],
    'deterministic_eval': [1, ],
    'normalize_advantage': [1, ],
}

reacher_configs = {
    'env_name': ['reacher'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [50_000_000, ],
    'episode_length': [200, 1_000],
    'action_repeat': [5, ],
    'num_envs': [2048],
    'num_eval_envs': [128],
    'learning_rate': [3e-4],
    'entropy_cost': [1e-3],
    'discounting': [0.95],
    'seed': list(range(5)),
    'unroll_length': [50, ],
    'batch_size': [256, ],
    'num_minibatches': [32, ],
    'num_updates_per_batch': [8, ],
    'num_evals': [20],
    'normalize_observations': [1, ],
    'reward_scaling': [5.0],
    'clipping_epsilon': [0.3],
    'gae_lambda': [0.95],
    'deterministic_eval': [1, ],
    'normalize_advantage': [1, ],
}

swimmer_configs = {
    'env_name': ['swimmer'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [50_000_000, ],
    'episode_length': [200],
    'action_repeat': [5, ],
    'num_envs': [2048],
    'num_eval_envs': [128],
    'learning_rate': [3e-4],
    'entropy_cost': [1e-3],
    'discounting': [0.95],
    'seed': list(range(5)),
    'unroll_length': [50, ],
    'batch_size': [256, ],
    'num_minibatches': [32, ],
    'num_updates_per_batch': [8, ],
    'num_evals': [20],
    'normalize_observations': [1, ],
    'reward_scaling': [5.0],
    'clipping_epsilon': [0.3],
    'gae_lambda': [0.95],
    'deterministic_eval': [1, ],
    'normalize_advantage': [1, ],
}


def main():
    command_list = []
    flags_combinations = dict_permutations(swimmer_configs)
    flags_combinations += dict_permutations(reacher_configs)
    # flags_combinations += dict_permutations(ant_configs)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[8],
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem=64000)


if __name__ == '__main__':
    main()
