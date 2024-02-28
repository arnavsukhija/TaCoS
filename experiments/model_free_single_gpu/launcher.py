import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'SingleGPUTestFeb28_14_00'

# general_configs = {
#     'env_name': ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum',
#                  'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d'],
#     'backend': ['generalized', ],
#     'project_name': [PROJECT_NAME],
#     'num_timesteps': [1_000_000, ],
# }


general_configs = {
    'env_name': ['ant', 'halfcheetah', 'hopper'],
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [1_000_000, ],
    'action_repeat': [2, 4, 5, 8, 10]
}


def main():
    command_list = []
    flags_combinations = dict_permutations(general_configs)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='3:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
