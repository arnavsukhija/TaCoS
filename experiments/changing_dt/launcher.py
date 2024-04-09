import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'HalfcheetahVaryingDtNoSwitchCostApr08_15_10'

# general_configs = {
#     'project_name': [PROJECT_NAME],
#     'backend': ['generalized', ],
#     'num_timesteps': [1_000_000, ],
#     'base_discount_factor': [0.99],
#     'num_envs': [128],
#     'num_env_steps_between_updates': [10, ],
#     'seed': list(range(5)),
#     'networks': [0, ],
#     'batch_size': [128],
#     'action_repeat': [1, ],
# }

# halfcheetah = {'env_name': ['halfcheetah', ],
#                'reward_scaling': [1.0, ],
#                'episode_time': [10.0],
#                'base_dt_divisor': [1, 2, 4, 10, 15],
#                'switch_cost_wrapper': [0, ]
#                } | general_configs

# halfcheetah_switch_cost = {'env_name': ['halfcheetah', ],
#                            'reward_scaling': [1.0, ],
#                            'episode_time': [10.0],
#                            'base_dt_divisor': [1, 2, 4, 10, 15],
#                            'switch_cost_wrapper': [1, ],
#                            'switch_cost': [0.01, 0.1, 0.2, 0.5, 1.0],
#                            'max_time_between_switches': [0.05, 0.1, 0.15, 0.2],
#                            'time_as_part_of_state': [0, 1]
#                            } | general_configs

halfcheetah_no_switch_cost_base_configs = {
    'project_name': [PROJECT_NAME],
    'env_name': ['halfcheetah', ],
    'reward_scaling': [1.0, ],
    'episode_time': [10.0],
    'switch_cost_wrapper': [0, ],
    'backend': ['generalized', ],
    'base_discount_factor': [0.99],
    'num_envs': [128],
    'num_env_steps_between_updates': [10, ],
    'seed': list(range(5)),
    'networks': [0, ],
    'batch_size': [128],
    'action_repeat': [1, ],
}

halfcheetah_no_switch_cost_configs = []
base_dt_divisor = [1, 2, 4, 10, 15]
base_numsteps = 1_000_000
for dt_divisor in base_dt_divisor:
    cur_configs = halfcheetah_no_switch_cost_base_configs | {'base_dt_divisor': [dt_divisor],
                                                             'num_timesteps': [base_numsteps * dt_divisor]}
    halfcheetah_no_switch_cost_configs.append(cur_configs)

for conf in halfcheetah_no_switch_cost_configs:
    print(f"Base dt divisor{conf['base_dt_divisor']}")
    print(f"Base numsteps{conf['num_timesteps']}")


def main():
    command_list = []
    flags_combinations = None
    for conf in halfcheetah_no_switch_cost_configs:
        if flags_combinations is None:
            flags_combinations = dict_permutations(conf)
        else:
            flags_combinations += dict_permutations(conf)
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
                          mem=32000)


if __name__ == '__main__':
    main()
