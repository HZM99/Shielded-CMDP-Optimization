import gymnasium as gym
import numpy as np
import time
import os
import logging
import argparse

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


from rl_constrained_smartgrid_control.environments.bus33_environment import IEEE33BusEnv
from rl_constrained_smartgrid_control.environments.bus69_environment import IEEE69BusEnv


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if "eq_viol" in info:
            self.logger.record("rollout/ep_eq_violation", np.sum(np.abs(info["eq_viol"])))
        if "ineq_viol" in info:
            self.logger.record("rollout/ep_ineq_violation", np.sum(np.maximum(0, info["ineq_viol"])))
        return True

def main():

    parser = argparse.ArgumentParser(description="Benchmark A2C on a specified bus system.")
    parser.add_argument("--bus", type=int, choices=[33, 69], required=True, help="The bus system to run (33 or 69).")
    args = parser.parse_args()

   
    if args.bus == 33:
        env_class = IEEE33BusEnv
        logging.info("Selected IEEE 33-Bus Environment.")
    elif args.bus == 69:
        env_class = IEEE69BusEnv
        logging.info("Selected IEEE 69-Bus Environment.")
    
    env = make_vec_env(lambda: env_class(), n_envs=1)

    
    log_dir = "tensorboard_logs/"
    run_name = f"A2C_{args.bus}Bus_{int(time.time())}"

    model = A2C("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log=log_dir)
    
    
    training_timesteps = 100_000 
    logging.info(f"--- Starting A2C Training on {args.bus}-Bus for {training_timesteps} timesteps ---")
    
    start_time = time.time()
    model.learn(total_timesteps=training_timesteps, callback=TensorboardCallback(), tb_log_name=run_name)
    end_time = time.time()

    logging.info(f"--- A2C Training Finished on {args.bus}-Bus ---")
    logging.info(f"Training took: {end_time - start_time:.2f} seconds")

   
    logging.info("--- Starting Final Evaluation ---")
    num_eval_episodes = 20

    total_rewards = []
    total_eq_violations = []
    total_ineq_violations = []
    for episode in range(num_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward, episode_eq_viol, episode_ineq_viol = 0, 0, 0
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
    logging.info(f"--- A2C {args.bus}-BUS FINAL EVALUATION RESULTS ---")
    logging.info(f"Average Reward: {avg_reward:.4f}")
    logging.info(f"Average Summed Equality Violation: {avg_eq_viol:.6f}")
    logging.info(f"Average Summed Inequality Violation: {avg_ineq_viol:.6f}")
    env.close()


if __name__ == "__main__":
    main()