import cloudpickle
import wandb
from jax._src.nn.functions import swish
from mujoco_playground._src import registry
import ast
from wtc.agents.ppo.ppo_brax_env import PPO
# Setup WandB API
api = wandb.Api()

# Replace with your actual project and run ID
run_path = "arnavsukhija-eth-zurich/PPOGo1JoystickFlatTerrain_fixObs/171bw71n"
run = wandb.init(project="arnavsukhija-eth-zurich/PPOGo1JoystickFlatTerrain_fixObs", id="171bw71n")
# Load config as a dictionary
config = dict(run.config)

# Optional: print out to verify
print("Loaded config from W&B run:", config)

# Load the policy parameters
with open('Policies/Policies/policy_params_171bw71n.pkl', 'rb') as f:
    policy_params = cloudpickle.load(f)

wandb.save(policy_params)

def parse_config(config):
    return {k: ast.literal_eval(v) if isinstance(v, str) else v for k, v in config.items()}

env_name = "Go1JoystickFlatTerrain"
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

config = parse_config(config)
randomization_fn = registry.get_domain_randomizer(env_name)
# Create the optimizer using W&B config
optimizer = PPO(
    environment=env,
    num_timesteps=config["num_timesteps"],
    episode_length=config["episode_length"],
    action_repeat=config["action_repeat"],
    num_envs=config["num_envs"],
    num_eval_envs=config["num_eval_envs"],
    lr=config["learning_rate"],
    wd=0.0,
    entropy_cost=config["entropy_cost"],
    unroll_length=config["unroll_length"],
    discounting=config["discounting"],
    batch_size=config["batch_size"],
    num_minibatches=config["num_minibatches"],
    num_updates_per_batch=config["num_updates_per_batch"],
    num_evals=config["num_evals"],
    normalize_observations=True,
    reward_scaling=config["reward_scaling"],
    max_grad_norm=config["max_grad_norm"],
    clipping_epsilon=0.3,
    gae_lambda=0.95,
    policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
    policy_activation=swish,  # If custom, make sure to define it
    critic_hidden_layer_sizes=config["critic_hidden_layer_sizes"],
    critic_activation=swish,  # Same here
    deterministic_eval=False,
    normalize_advantage=True,
    wandb_logging=True,
    randomization_fn=randomization_fn,  # If applicable
    policy_obs_key=config["policy_obs_key"],
    value_obs_key=config["value_obs_key"],
    seed=config["seed"]
)
policy = optimizer.make_policy(policy_params, deterministic=True)
