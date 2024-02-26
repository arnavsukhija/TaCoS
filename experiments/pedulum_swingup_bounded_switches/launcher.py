import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'PendulumSwingUpBoundedSwitchesFeb26_12_19'

general_configs = {
    'project_name': [PROJECT_NAME],
    'episode_length': [200, ],
    'learning_discount_factor': [0.99, ],
    'min_reps': [0, ],
    'max_reps': [50, ],
    'sac_train_steps': [200_000, ],
    'wandb_logging': [True, ],
    'plot_progress': [False, ],
    'training_seed': list(range(5)),
}

configs_wrapper = {'wrapper': [True, ],
                   'num_switches': list(range(10, 31)),
                   } | general_configs
configs_action_repeat = {'wrapper': [False, ],
                         'action_repeat': [4, 5, 8, 10, 20],
                         } | general_configs


def main():
    command_list = []

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
