import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'SAC_Apr15_13_40'

swimmer_configs = {
    'env_name': ['swimmer'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [1_000_000, ],
    'episode_length': [200],
    'num_envs': [128],
    'learning_discount_factor': [0.99],
    'num_env_steps_between_updates': [10, ],
    'seed': list(range(5)),
    'networks': [0],
    'batch_size': [128],
    'action_repeat': [1, ],
    'reward_scaling': [1.0, ],
}

hopper_configs = {
    'env_name': ['hopper'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [1_000_000, ],
    'episode_length': [200],
    'num_envs': [128],
    'learning_discount_factor': [0.99],
    'num_env_steps_between_updates': [10, ],
    'seed': list(range(5)),
    'networks': [0],
    'batch_size': [128],
    'action_repeat': [1, ],
    'reward_scaling': [30.0, ],
}


def main():
    command_list = []
    flags_combinations = dict_permutations(swimmer_configs)
    flags_combinations += dict_permutations(hopper_configs)

    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
