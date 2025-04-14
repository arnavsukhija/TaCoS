import wandb
import pandas as pd
import re

# Initialize wandb API
api = wandb.Api()

# Set your entity ant project name
entity = "trevenl"
project = "NoiseInfluence_Apr_16_11_45"

# Fetch all runs from the project
runs = api.runs(f"{entity}/{project}")
# Create an empty list to hold data
data = []

# Loop through runs ant collect data
for index, run in enumerate(runs):
    print(f'Starting run {index}')
    # Example of fetching run name, config, summary metrics
    cost_value = None
    pattern = r"--switch_cost=(\d+\.?\d*)"
    try:
        for item in run.metadata['args']:
            match = re.search(pattern, item)
            if match:
                cost_value = float(match.group(1))
                break
    except:
        print('Extraction of switch cost not possible')
    run_data = {
        "name": run.name,
        "config": run.config,
        "summary": run.summary._json_dict,
        "switch_cost": cost_value
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

# You can now save this DataFrame to a CSV file or perform further analysis
df.to_csv("data/noise_influence_performance.csv", index=False)
