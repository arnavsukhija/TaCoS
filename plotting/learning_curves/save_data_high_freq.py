import wandb
import pandas as pd

# Initialize wandb API
api = wandb.Api()

# Set your entity ant project name
entity = "trevenl"

env = 'reacher'
BASE_DT_DIVISOR = 10
files = {
    'high_freq_sac_tac': 'ReacherSwitchCostApr24_09_45',
    'high_freq_sac': "ReacherNoSwitchCostApr24_10_00",
}

# env = 'humanoid'
# BASE_DT_DIVISOR = 5
# files = {
#     'high_freq_sac_tac': 'HumanoidSwitchCostMay10_17_45',
#     'high_freq_sac': "HumanoidNoSwitchCostMay10_17_45",
# }

# env = 'rccar'
# BASE_DT_DIVISOR = 80
# files = {
#     'high_freq_sac_tac': 'RCCarSwitchCostApr16_15_00',
#     'high_freq_sac': "RCCarNoSwitchCostApr16_15_00",
# }


# env = 'halfcheetah'
# BASE_DT_DIVISOR = 10
# files = {
#     'high_freq_sac_tac': 'HalfcheetahSwitchCostApr11_10_00',
#     'high_freq_sac': "HalfcheetahNoSwitchCostApr12_14_00",
# }


# logging_data = 'eval_true_env/episode_reward'
length_key = 'eval/avg_episode_length'
reward_key = 'eval/episode_reward'


for baseline, project_name in files.items():
    runs = api.runs(f"{entity}/{project_name}")
    data = []
    # Loop through runs ant collect data
    for idx, run in enumerate(runs):
        print(f"Run {idx + 1}")
        if run.config.get('base_dt_divisor', -1) == BASE_DT_DIVISOR :
            # Example of fetching run name, config, summary metrics
            history = run.scan_history(keys=[length_key, reward_key])

            # Save the data of each run, depending on your plotting needs, may need further processing
            reward_performance = [(item[length_key], item[reward_key]) for item in history if
                                  length_key in item and reward_key in item]
            plot_tuple = list(zip(*reward_performance))

            run_data = {
                "name": run.name,
                "config": run.config,
                "summary": run.summary._json_dict,
                "lengths_and_rewards": plot_tuple,
            }
            data.append(run_data)
        else:
            print(f"Skipping min time repeat: {run.config.get('base_dt_divisor', -1)}")

    # Convert list into pandas DataFrame
    df = pd.DataFrame(data)

    # Optional: Expand the config ant summary dicts into separate columns
    config_df = df['config'].apply(pd.Series)
    summary_df = df['summary'].apply(pd.Series)

    # Joining the expanded config ant summary data with the original DataFrame
    df = df.join(config_df).join(summary_df)

    print(df.head())  # Display the first few rows of the DataFrame

    # You can now save this DataFrame to a CSV file or perform further analysis
    df.to_csv(f"data/{env}/{baseline}.csv", index=False)
