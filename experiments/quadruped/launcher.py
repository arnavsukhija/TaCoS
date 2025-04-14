from experiments.util import generate_base_command, generate_run_commands, available_gpus
import training


def main():
    command_list = [generate_base_command(training)]
    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[3],
                          mode='euler',
                          duration='03:59:00',
                          prompt=True,
                          mem=32000)


if __name__ == '__main__':
    main()