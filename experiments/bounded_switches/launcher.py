import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'GreenhouseBoundedSwitchesApr22_15_00'

general_configs = {
    'project_name': [PROJECT_NAME],
    'env_name': ['GreenHouse', ],
    'sac_train_steps': [200_000, ],
    'training_seed': list(range(5)),
    'plot_progress': [0, ],
    'episode_length': [300, ],
}

bounded_switches_config = {'wrapper': [1, ],
                           'action_repeat': [1, ],
                           'num_switches': list(range(3, 21)) + [25, 30, 35],
                           } | general_configs

action_repeat_configs = {'wrapper': [0, ],
                         'num_switches': [-1, ],
                         'action_repeat': [4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100, 150],
                         } | general_configs


def main():
    command_list = []
    flags_combinations = dict_permutations(bounded_switches_config) + dict_permutations(action_repeat_configs)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=0,
                          mode='euler',
                          duration='3:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
