import exp
import numpy as np
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'NoiseInfluencePerformance_Apr_04_10_00'

general_configs = {
    'env_name': ['Greenhouse', 'Pendulum'],
    'project_name': [PROJECT_NAME, ],
    'scale': list(np.linspace(0, 2, 40)),
    'seed': list(range(5)),
    'wrapper': [0, ],
    'num_timesteps': [100_000],
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
