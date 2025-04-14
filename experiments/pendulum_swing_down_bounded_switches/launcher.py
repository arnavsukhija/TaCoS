import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'PendulumSwingDownBoundedSwitchesFeb26_16_37'

general_configs = {
    'project_name': [PROJECT_NAME],
    'episode_length': [300, ],
    'learning_discount_factor': [0.99, ],
    'min_reps': [1, ],
    'max_reps': [60, ],
    'sac_train_steps': [200_000, ],
    'wandb_logging': [1, ],
    'plot_progress': [0, ],
    'training_seed': list(range(5)),
}

configs_wrapper = {'wrapper': [1, ],
                   'num_switches': list(range(3, 21)) + [25, 30, 35],
                   } | general_configs

configs_action_repeat = {'wrapper': [0, ],
                         'action_repeat': [4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100],
                         } | general_configs


def main():
    command_list = []

    # flags_combinations = dict_permutations(configs_action_repeat)
    # flags_combinations = dict_permutations(configs_wrapper)
    flags_combinations = dict_permutations(configs_wrapper) + dict_permutations(configs_action_repeat)
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
