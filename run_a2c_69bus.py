import gymnasium as gym
import numpy as np
import time
import os
import logging

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from rl_constrained_smartgrid_control.environments.bus69_environment import IEEE69BusEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        
        if "eq_viol" in info:
            eq_violation = np.sum(np.abs(info["eq_viol"]))
            self.logger.record("rollout/ep_eq_violation", eq_violation)
        if "ineq_viol" in info:
            ineq_violation = np.sum(np.maximum(0, info["ineq_viol"]))
            self.logger.record("rollout/ep_ineq_violation", ineq_violation)
            
        return True

def main():
    env = make_vec_env(lambda: IEEE69BusEnv(), n_envs=1)
    
    log_dir = "tensorboard_logs/"
    run_name = f"A2C_69Bus_{int(time.time())}"

    model = A2C("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log=log_dir)
    
    training_timesteps = 100_000
    logging.info(f"--- Starting A2C Training on 69-Bus for {training_timesteps} timesteps ---")
    
    start_time = time.time()
    model.learn(total_timesteps=training_timesteps, callback=TensorboardCallback(), tb_log_name=run_name)
    end_time = time.time()

    logging.info("--- A2C Training Finished ---")
    logging.info(f"Training took: {end_time - start_time:.2f} seconds")

    logging.info("--- Starting Final Evaluation ---")
    num_eval_episodes = 20
    total_rewards = []
    total_eq_violations = []
    total_ineq_violations = []

    for episode in range(num_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_eq_viol = 0
        episode_ineq_viol = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            if "eq_viol" in info[0]:
                episode_eq_viol += np.sum(np.abs(info[0]["eq_viol"]))
            if "ineq_viol" in info[0]:
                episode_ineq_viol += np.sum(np.maximum(0, info[0]["ineq_viol"]))

        total_rewards.append(episode_reward)
        total_eq_violations.append(episode_eq_viol)
        total_ineq_violations.append(episode_ineq_viol)
        logging.info(f"Eval Episode {episode + 1}/{num_eval_episodes} -> Reward: {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    avg_eq_viol = np.mean(total_eq_violations)
    avg_ineq_viol = np.mean(total_ineq_violations)

    logging.info("--- A2C 69-BUS FINAL EVALUATION RESULTS ---")
    logging.info(f"Average Reward: {avg_reward:.4f}")
    logging.info(f"Average Summed Equality Violation: {avg_eq_viol:.6f}")
    logging.info(f"Average Summed Inequality Violation: {avg_ineq_viol:.6f}")
    
    env.close()

if __name__ == "__main__":
    main()