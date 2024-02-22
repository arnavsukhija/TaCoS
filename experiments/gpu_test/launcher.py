import exp
from experiments.util import generate_run_commands, generate_base_command



def main():
    command_list = []

    cmd = generate_base_command(exp, flags=None)
    command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=2, num_gpus=4, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()