# Save this file as: train_shielded_ppo.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # <-- 1. IMPORT THE MONITOR

# 1. Import your environment and your new wrapper
from rl_constrained_smartgrid_control.environments.bus33_environment import IEEE33BusEnv
from shield_wrapper import SafetyShieldWrapper

# --- This is your main script ---

print("Starting Shielded PPO (Stable Baselines3) training...")

# 1. Create your base 33-bus environment
base_env = IEEE33BusEnv()

# 2. Wrap the base environment with your new Safety Shield
safe_env = SafetyShieldWrapper(base_env)

# 3. --- ADD THE MONITOR WRAPPER ---
# This is the "scoreboard" that records episode rewards
monitored_safe_env = Monitor(safe_env)
# ---------------------------------

# 4. Vectorize the environment (SB3 requires this)
# We now pass the *monitored* environment
vec_safe_env = DummyVecEnv([lambda: monitored_safe_env]) 

# 5. Create your PPO agent from Stable Baselines3
print("Creating Stable Baselines3 PPO model...")
model = PPO(
    "MlpPolicy",
    vec_safe_env,
    verbose=1,
    tensorboard_log="./ppo_shielded_tensorboard/", # Logger is still here
)

# 6. Train as normal
print("Training agent...")
model.learn(total_timesteps=2_000_000) # Quick run to get graphs

print("Training complete.")