import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'NormalTraining_Mar_27_10_30'

general_configs = {
    'project_name': [PROJECT_NAME],
    'num_timesteps': [5_000_000, ],
    'episode_length': [500],
    'learning_discount_factor': [0.99],
    'num_envs': [128, 264],
    'num_env_steps_between_updates': [1, 10, 20, ],
    'seed': list(range(3)),
    'networks': [0, ],
    'batch_size': [128, 512],
    'action_repeat': [1, 2, 3, 10],
}

ant = {'env_name': ['ant', ],
       'reward_scaling': [10, ],
       } | general_configs


def main():
    command_list = []
    flags_combinations = dict_permutations(ant)
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
