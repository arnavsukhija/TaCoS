import wandb

# Set your W&B entity and project
ENTITY = "arnavsukhija-eth-zurich"
PROJECT = "PPO_framestack3_Mar29_noDRsampling_75mil"
NEW_PARAM_KEY = "action_delay"
NEW_PARAM_VALUE = 2.0

# Initialize W&B API
api = wandb.Api()

# Get all runs in the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in runs:
    print(f"Updating run: {run.id}")

    # Get the current config
    config = run.config
    config[NEW_PARAM_KEY] = NEW_PARAM_VALUE  # Add new parameter

    # Update the run with the new config
    run.update()  # W&B does not allow direct config updates, but you can log a new config
    wandb.init(id=run.id, project=PROJECT, entity=ENTITY, resume="allow")
    wandb.config.update(config)
    wandb.finish()

    print(f"Updated run {run.id} with {NEW_PARAM_KEY}={NEW_PARAM_VALUE}")