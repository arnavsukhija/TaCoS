import pickle
import os
import wandb
from wtc.envs.rccar import RCCar, plot_rc_trajectory

def experiment(env_name: str = 'inverted_pendulum',
               backend: str = 'mjx',
               filename: str = None,
               track: bool = False,
               dir: str = 'random',
               plot: bool = False,
               ):
    assert env_name in ['rccar']
    assert backend in ['generalized', 'positional', 'spring', 'mjx']
    env = RCCar(margin_factor=20)

    with open(os.path.join(dir, filename), 'rb') as fp:
        trajectory = pickle.load(fp)

    fig, axs = plot_rc_trajectory(trajectory.obs)
    print('Done')


def get_all_runs(project_name, entity_name):
    """
    Retrieves and prints configurations from all runs in a WandB project.

    Args:
        project_name (str): The name of your WandB project.
        entity_name (str): The name of your WandB entity (username or team).
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity_name}/{project_name}")

        if not runs:
            print(f"No runs found in {entity_name}/{project_name}.")
            return

        for run in runs:
            print(f"\nRun ID: {run.id}")
            if run.config:
                print("  Configuration:")
                for key, value in run.config.items():
                    print(f"    {key}: {value}")
            else:
                print("  No configuration found.")

        return True  # Success

    except wandb.CommError as e:
        print(f"Error connecting to WandB: {e}")
        return False

    except wandb.errors.CommError as e:
        print(f"Error connecting to WandB: {e}")
        return False

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == '__main__':
    api = wandb.Api()
    projects = api.projects(entity='arnavsukhija-eth-zurich')
    for p in projects:
        print(p.name)
    print(get_all_runs('TaCoSPPO_2Actions_Mar16', 'arnavsukhija-eth-zurich'))