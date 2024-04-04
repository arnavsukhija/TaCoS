import wandb
import pandas as pd

# Initialize wandb API
api = wandb.Api()

# Set your entity and project name
entity = "trevenl"
project = "PendulumSwingUpBoundedSwitchesFeb26_13_14"

# Fetch all runs from the project
runs = api.runs(f"{entity}/{project}")

# Create an empty list to hold data
data = []

# Loop through runs and collect data
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

# Optional: Expand the config and summary dicts into separate columns
config_df = df['config'].apply(pd.Series)
summary_df = df['summary'].apply(pd.Series)

# Joining the expanded config and summary data with the original DataFrame
df = df.join(config_df).join(summary_df)

print(df.head())  # Display the first few rows of the DataFrame

# You can now save this DataFrame to a CSV file or perform further analysis
df.to_csv("data/pendulum_data.csv", index=False)