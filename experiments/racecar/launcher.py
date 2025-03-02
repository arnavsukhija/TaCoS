import exp
import evaluation_script
from experiments.util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

################################################
#################### RC Car ####################
################################################

rccar_switch_cost_control_interval_base_config = {
    'env_name': ['rccar', ],
    'backend': ['generalized', ],
    'project_name': ["RCCarPPOSwitchCostControlInterval_Feb25_11_00"],
    'num_timesteps': [2_000_000, ],
    'episode_time': [4.0, ],
    'base_dt_divisor': [1,],
    'base_discount_factor': [0.9],
    'seed': list(range(5)),
    'num_envs': [2048],
    'num_eval_envs': [32],
    'entropy_cost': [1e-2],
    'unroll_length': [10],
    'num_minibatches': [32],
    'num_updates_per_batch': [4],
    'batch_size': [1024],
    'networks': [0, ],
    'reward_scaling': [1.0, ],
    'switch_cost_wrapper': [1, ],
    'switch_cost': [0.1, ],
    'time_as_part_of_state': [1, ],
    'num_final_evals': [1, ]
}

# Lenart's version
# rccar_switch_cost_low_freq_base = {'env_name': ['rccar', ],
#                                    'backend': ['generalized', ],
#                                    'project_name': ["RCCARPPOSwitchCostMay14_16_05"],
#                                    'num_timesteps': [2_000_000, ],
#                                    'episode_time': [4.0, ],
#                                    'base_dt_divisor': [1, ], #base_dt_divisor 1 enforces env.dt to be 0.5
#                                    'base_discount_factor': [0.9],
#                                    'seed': list(range(5)),
#                                    'num_envs': [2048],
#                                    'num_eval_envs': [32],
#                                    'entropy_cost': [1e-2],
#                                    'unroll_length': [10],
#                                    'num_minibatches': [32],
#                                    'num_updates_per_batch': [4],
#                                    'batch_size': [1024],
#                                    'networks': [0, ],
#                                    'reward_scaling': [1.0, ],
#                                    'switch_cost_wrapper': [1, ],
#                                    'switch_cost': [0.1, ],
#                                    'time_as_part_of_state': [1, ],
#                                    'num_final_evals': [1, ]
#                                    }
#
# rccar_switch_cost_low_freq = []
# min_time_multipliers = [2, 3, 5, ]
# for min_time_multiplier in min_time_multipliers:
#     cur_configs = rccar_switch_cost_low_freq_base | {'min_time_repeat': [min_time_multiplier],
#                                                      'max_time_between_switches': [min_time_multiplier * 0.5]} #this is the same as min_time_between_switches (so tmin=tmax), see exp.py optimizer
#     rccar_switch_cost_low_freq.append(cur_configs)

rccar_switch_cost = {'env_name': ['rccar', ],
                     'backend': ['generalized', ],
                     'project_name': ["TaCoSPPO_RCCar_Eval_Mar02_17_00"],
                     'num_timesteps': [2_000_000, ],
                     'episode_steps': [200, ],
                     'base_discount_factor': [0.9],
                     'seed': list(range(5)),
                     'num_envs': [2048],
                     'num_eval_envs': [32],
                     'entropy_cost': [1e-2],
                     'unroll_length': [10],
                     'num_minibatches': [32],
                     'num_updates_per_batch': [4],
                     'batch_size': [1024],
                     'networks': [0, ],
                     'reward_scaling': [1.0, ],
                     'switch_cost': [0.1, ],
                     'max_time_repeat': [10],
                     'time_as_part_of_state': [1, ],
                     'num_final_evals': [10, ],
                     }

rccar_no_switch_cost_ppo = {'env_name': ['rccar', ],
                     'backend': ['generalized', ],
                     'project_name': ["TaCoSPPO_RCCar_Eval_Mar02_14_00"],
                     'num_timesteps': [50_000_000, ],
                     'episode_steps': [200, ],
                     'base_discount_factor': [0.9],
                     'seed': list(range(5)),
                     'num_envs': [2048],
                     'num_eval_envs': [32],
                     'entropy_cost': [1e-2],
                     'unroll_length': [10],
                     'num_minibatches': [32],
                     'num_updates_per_batch': [4],
                     'batch_size': [1024],
                     'networks': [0, ],
                     'reward_scaling': [1.0, ],
                     'switch_cost': [0.1, ],
                     'max_time_repeat': [10],
                     'time_as_part_of_state': [1, ],
                     'num_final_evals': [10, ],
                     }


def main():
    command_list = []
    flags_combinations = dict_permutations(rccar_switch_cost)
    for flags in flags_combinations:
        cmd = generate_base_command(evaluation_script, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[7],
                          mode='euler',
                          duration='03:59:00',
                          prompt=True,
                          mem=32000)
if __name__ == '__main__':
    main()
