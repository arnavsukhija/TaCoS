import wandb
import numpy as np
# Initialize wandb API
api = wandb.Api()

# Set your entity ant project name
entity = "trevenl"
project = 'Model-Free-Tacos'
# run_id = 'opay4fyf'
run_id = 'bo2xs24h'



# EPISODE_TIME = 4.0
# STEPS_PER_UPDATE = 20_000 / 20

EPISODE_TIME = 10.0
STEPS_PER_UPDATE = 100_000 / 20

# logging_data = 'eval_true_env/episode_reward'
length_key = 'eval/avg_episode_length'
reward_key = 'eval/episode_reward'


run = api.run(f"{entity}/{project}/{run_id}")
# Example of fetching run name, config, summary metrics
history = run.scan_history(keys=[length_key, reward_key])

# Save the data of each run, depending on your plotting needs, may need further processing
num_steps = [item[length_key] for item in history if
                      length_key in item and reward_key in item]


num_steps = np.array(num_steps)
xs = STEPS_PER_UPDATE / num_steps * EPISODE_TIME
xs = np.cumsum(xs)
print(xs[-1] / EPISODE_TIME)


