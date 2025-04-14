import exp
import numpy as np
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'SwitchCostPerformance_Apr15_16_00'

general_configs = {
    'env_name': ['Greenhouse', ],
    'project_name': [PROJECT_NAME, ],
    'switch_cost': list(np.linspace(5, 15, 50)),
    'seed': list(range(5)),
    'wrapper': [1, ],
    'num_timesteps': [200_000],
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
