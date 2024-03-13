import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'BoundedSwitches_Feb_29_14_00'

general_configs = {
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [1_000_000, ],
    'episode_length': [1000, ],
    'learning_discount_factor': [0.99],
    'min_reps': [1],
    'max_reps': [10, ],
    'seed': list(range(5))
}

hopper = {'env_name': ['hopper',],
          'num_switches': [150,],
          } | general_configs

halfcheetah = {'env_name': ['halfcheetah',],
               'num_switches': [250,],
               } | general_configs


def main():
    command_list = []
    flags_combinations = dict_permutations(hopper) + dict_permutations(halfcheetah)
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
