import wandb
import pandas as pd
import os

# Initialize wandb API
api = wandb.Api()

# Set your entity ant project name
entity = "trevenl"

############### Reacher 4.0 sec horizon ###############
# env_name = 'reacher'
# runs_to_download = {"switch_cost": 'ReacherPPOSwitchCostMay14_14_20',
#                     'low_freq': "ReacherPPOSwitchCostLowFreqMay20_09_20"
# }


############### Reacher 2.0 sec horizon ###############
# env_name = 'reacher'
# runs_to_download = {"no_switch_cost": 'ReacherNoSwitchCostApr24_10_00',
#                     "switch_cost": 'ReacherSwitchCostApr24_09_45',
#                     'naive_model': 'ReacherNoSwitchCostMay08_15_45'}


############### RCCar 4.0 sec horizon ###############
# env_name = 'rccar'
# runs_to_download = {
#     'switch_cost': 'RCCARPPOSwitchCostMay14_16_05',
#     'low_freq': "RCCARPPOSwitchCostMay14_16_05"
# }

############### Humanoid 3.0 sec horizon ###############
# env_name = 'humanoid'
# runs_to_download = {
#     "switch_cost": 'HumanoidPPOSwitchCostMay14_14_20',
#     "low_freq": 'HumanoidPPOSwitchCostLowFreqMay20_09_20'
# }

############### Halfcheetah 10.0 sec horizon ###############
env_name = 'halfcheetah'
runs_to_download = {"switch_cost": 'HalfcheetahPPOSwitchCostMay20_15_00',
                    'low_freq': 'HalfcheetahPPOSwitchCostLowFreqMay20_15_00'
                    }

# Fetch all runs from the project
for filename, run_name in runs_to_download.items():
    runs = api.runs(f"{entity}/{run_name}")

    # Create an empty list to hold data
    data = []

    # Loop through runs ant collect data
    for run in runs:
        # Example of fetching run name, config, summary metrics
        run_data = {
            "name": run.name,
            "config": run.config,
            "summary": run.summary._json_dict,
        }
        data.append(run_data)

    # Convert list into pandas DataFrame
    df = pd.DataFrame(data)

    # Optional: Expand the config ant summary dicts into separate columns
    config_df = df['config'].apply(pd.Series)
    summary_df = df['summary'].apply(pd.Series)

    # Joining the expanded config ant summary data with the original DataFrame
    df = df.join(config_df).join(summary_df)

    print(df.head())  # Display the first few rows of the DataFrame

    directory = os.path.join("data", env_name)

    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

    # You can now save this DataFrame to a CSV file or perform further analysis
    df.to_csv(os.path.join(directory, f"ppo_{filename}.csv"), index=False)
    # df.to_csv("data/reacher/no_switch_cost.csv", index=False)
