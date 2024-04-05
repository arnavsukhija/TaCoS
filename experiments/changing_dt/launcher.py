import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'HalfcheetahVaryingDtApr05_12_55'

general_configs = {
    'project_name': [PROJECT_NAME],
    'backend': ['generalized', ],
    'num_timesteps': [1_000_000, ],
    'base_discount_factor': [0.99],
    'num_envs': [128],
    'num_env_steps_between_updates': [10, ],
    'seed': [0],
    'networks': [0, ],
    'batch_size': [128],
    'action_repeat': [1, ],
}

inverted_pendulum = {'env_name': ['inverted_pendulum', ],
                     'reward_scaling': [1.0, ],
                     'episode_time': [8.0],
                     'base_dt_divisor': [1, 2, 4, ],
                     } | general_configs

halfcheetah = {'env_name': ['halfcheetah', ],
               'reward_scaling': [1.0, ],
               'episode_time': [10.0],
               'base_dt_divisor': [1, 2, 4, 10, 15, 20, 30, 50],
               } | general_configs


def main():
    command_list = []
    flags_combinations = dict_permutations(halfcheetah)
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
