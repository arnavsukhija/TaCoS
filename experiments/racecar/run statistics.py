import wandb
import pandas as pd

# Initialize API
api = wandb.Api()

# Set your project details
ENTITY = "arnavsukhija-eth-zurich"  # Replace with your W&B username or team
PROJECT = "TacoSHardwareSetup_noDRsampling_20Mil"  # Replace with your W&B project name
TAG = "hardware_actions4"  # Replace with the tag you are filtering for

# Get runs with the specified tag
runs = api.runs(f"{ENTITY}/{PROJECT}", {"tags": TAG})

# Extract metrics into a DataFrame
data = []
for run in runs:
    summary = run.summary._json_dict  # Get summary metrics
    name = run.name  # Run name

    # Append relevant information (modify as needed)
    data.append({"name": name, **summary})

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute descriptive statistics
stats = df.describe()

# Save results
df.to_csv(f"wandb_runs_{PROJECT}_{TAG}.csv", index=False)
stats.to_csv(f"wandb_stats_{PROJECT}_{TAG}.csv")

# Print summary
print(stats)
