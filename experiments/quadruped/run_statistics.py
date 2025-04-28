import wandb
import pandas as pd

# Initialize API
api = wandb.Api()

# Set your project details
ENTITY = "arnavsukhija-eth-zurich"  # Replace with your W&B username or team
PROJECT = "PPOGo1JoystickFlatTerrain_DR_fixedRewards"  # Replace with your W&B project name
TAG = "tacos10_hardware"  # Replace with the tag you are filtering for

# Get runs with the specified tag
runs = api.runs(f"{ENTITY}/{PROJECT}")

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
df.to_csv(f"wandb_runs_{PROJECT}.csv", index=False)
stats.to_csv(f"wandb_stats_{PROJECT}.csv")

# Print summary
print(stats)