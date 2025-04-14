import wandb
import pandas as pd

# Initialize wandb API
api = wandb.Api()

# Set your entity ant project name
entity = "trevenl"

env = 'reacher'
files = {
    'high_freq_sac': 'RCCarSACNoSwitchCostLowFreqMay20_13_30',
    'low_freq_sac_tac': "RCCarSACSwitchCostLowFreqMay20_10_15",
    'low_freq_ppo_tac': 'RCCARPPOSwitchCostMay14_16_05'
}

# env = 'rccar'
# files = {
#     'low_freq_sac': 'RCCarSACNoSwitchCostLowFreqMay20_13_30',
#     'low_freq_sac_tac': "RCCarSACSwitchCostLowFreqMay20_10_15",
#     'low_freq_ppo_tac': 'RCCARPPOSwitchCostMay14_16_05'
# }

# logging_data = 'eval_true_env/episode_reward'
length_key = 'eval/avg_episode_length'
reward_key = 'eval/episode_reward'

MIN_TIME_REPEAT = 2

for baseline, project_name in files.items():
    runs = api.runs(f"{entity}/{project_name}")
    data = []
    # Loop through runs ant collect data
    for idx, run in enumerate(runs):
        print(f"Run {idx + 1}")
        if run.config.get('min_time_repeat', -1) == MIN_TIME_REPEAT or run.config.get('action_repeat',
                                                                                      -1) == MIN_TIME_REPEAT:
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
            print(f"Skipping min time repeat: {run.config.get('min_time_repeat', -1)}")

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
