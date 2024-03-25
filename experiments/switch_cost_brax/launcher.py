import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'SwitchCost_Mar_25_12_00'

general_configs = {
    'backend': ['generalized', ],
    'project_name': [PROJECT_NAME],
    'num_timesteps': [1_000_000, ],
    'episode_length': [500, ],
    'learning_discount_factor': [0.99],
    'num_envs': [64, ],
    'num_env_steps_between_updates': [20, ],
    'seed': list(range(5)),
    'networks': [0, ]
}

hopper = {'env_name': ['hopper', ],
          'switch_cost': [0.1, ],
          'min_reps': [1],
          'max_reps': [30, ],
          } | general_configs

halfcheetah = {'env_name': ['halfcheetah', ],
               'switch_cost': [0.1, ],
               'min_reps': [1],
               'max_reps': [30, ],
               } | general_configs

ant = {'env_name': ['ant', ],
       'switch_cost': [0.1, ],
       'min_reps': [1],
       'max_reps': [20, ],
       } | general_configs

humanoid = {'env_name': ['humanoid', ],
            'switch_cost': [0.1, ],
            'min_reps': [1],
            'max_reps': [20, ],
            } | general_configs


def main():
    command_list = []
    flags_combinations = dict_permutations(hopper) + dict_permutations(halfcheetah)
    flags_combinations += dict_permutations(ant) + dict_permutations(humanoid)
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
