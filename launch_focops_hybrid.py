import os
import argparse
import gymnasium as gym
from omnisafe.algorithms.on_policy.first_order.focops import FOCOPS
from omnisafe.utils.config import Config
from omnisafe.envs.core import CMDP, env_register
from custom_logger import CustomSafetyLogger

# Import the existing Shield wrapper
from shield_wrapper import SafetyShieldWrapper

# Import BOTH base Wrappers
try:
    from ieee33_wrapper import IEEE33Wrapper
    from ieee69_wrapper import IEEE69Wrapper
except ImportError:
    print("Error: Could not import wrappers. Make sure ieee33_wrapper.py and ieee69_wrapper.py exist.")
    exit()

# Global custom logger instance
_custom_logger = None

# --- 33-Bus Hybrid Wrapper ---
@env_register
class IEEE33HybridWrapper(IEEE33Wrapper):
    _support_envs = ['IEEE33-Hybrid-v0']
    need_time_limit_wrapper = True
    need_auto_reset_wrapper = True
    
    def __init__(self, env_id: str, num_envs: int = 1, device: str = 'cpu', **kwargs):
        super().__init__(env_id, num_envs, device, **kwargs)
        print("\n🛡️  [33-BUS] HYBRID MODE: ACTIVATING SAFETY SHIELD  🛡️")
        # Import shield check function
        from shield_model import check_action_safety
        self.shield_check = check_action_safety
        self.shield_interventions = 0
        self.total_steps = 0
        # Episode-level tracking for custom metrics
        self.ep_ineq_violations = 0.0
        self.ep_shield_interventions = 0
        self.ep_steps = 0
    
    def step(self, action):
        """Override step to apply shield after action denormalization."""
        import numpy as np
        import torch
        
        # Denormalize action from [-1, 1] to original action space
        action_original = self._denormalize_action(action)
        
        # APPLY SHIELD: Check and correct the denormalized action
        is_safe, corrected_action = self.shield_check(
            self._env.state if hasattr(self._env, 'state') else None,
            action_original,
            self._env
        )
        
        # Track shield interventions
        if not is_safe:
            self.shield_interventions += 1
        self.total_steps += 1
        
        # Step with the shielded action
        obs, reward, terminated, truncated, info = self._env.step(corrected_action)
        
        # Calculate cost from violations
        eq_viol = info.get('eq_viol', 0.0)
        ineq_viol = info.get('ineq_viol', 0.0)
        
        if isinstance(eq_viol, (np.ndarray, list)):
            eq_cost = float(np.sum(eq_viol))
        else:
            eq_cost = float(eq_viol)
            
        if isinstance(ineq_viol, (np.ndarray, list)):
            ineq_cost = float(np.sum(ineq_viol))
        else:
            ineq_cost = float(ineq_viol)
        
        cost_float = eq_cost + ineq_cost
        info['cost'] = cost_float
        info['shield_intervention'] = not is_safe
        info['shield_rate'] = self.shield_interventions / max(1, self.total_steps)
        
        # Track episode-level custom metrics
        self.ep_ineq_violations += ineq_cost
        if not is_safe:
            self.ep_shield_interventions += 1
        self.ep_steps += 1
        
        # At episode end, log custom metrics with special prefix for TensorBoard
        if terminated or truncated:
            info['Metrics/EpisodeIneqViolations'] = self.ep_ineq_violations
            info['Metrics/EpisodeShieldInterventions'] = self.ep_shield_interventions
            info['Metrics/ShieldInterventionRate'] = self.ep_shield_interventions / max(1, self.ep_steps)
            
            # Log directly to custom TensorBoard logger
            global _custom_logger
            if _custom_logger is not None:
                _custom_logger.log_step(info)
            
            # Reset episode counters
            self.ep_ineq_violations = 0.0
            self.ep_shield_interventions = 0
            self.ep_steps = 0
        
        # Convert to tensors
        observation = torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        cost = torch.tensor(cost_float, dtype=torch.float32)
        terminated = torch.tensor(bool(terminated), dtype=torch.bool)
        truncated = torch.tensor(bool(truncated), dtype=torch.bool)
        info['original_reward'] = reward.clone()
        
        return observation, reward, cost, terminated, truncated, info

# --- 69-Bus Hybrid Wrapper (NEW) ---
@env_register
class IEEE69HybridWrapper(IEEE69Wrapper):
    _support_envs = ['IEEE69-Hybrid-v0']
    need_time_limit_wrapper = True
    need_auto_reset_wrapper = True
    
    def __init__(self, env_id: str, num_envs: int = 1, device: str = 'cpu', **kwargs):
        super().__init__(env_id, num_envs, device, **kwargs)
        print("\n🛡️  [69-BUS] HYBRID MODE: ACTIVATING SAFETY SHIELD  🛡️")
        # Import shield check function
        from shield_model import check_action_safety
        self.shield_check = check_action_safety
        self.shield_interventions = 0
        self.total_steps = 0
        # Episode-level tracking for custom metrics
        self.ep_ineq_violations = 0.0
        self.ep_shield_interventions = 0
        self.ep_steps = 0
        # Debug counter
        self.step_call_count = 0
    
    def step(self, action):
        """Override step to apply shield after action denormalization."""
        import numpy as np
        import torch
        
        # Debug: verify this method is being called
        self.step_call_count += 1
        if self.step_call_count == 1:
            print("✅ IEEE69HybridWrapper.step() IS BEING CALLED")
        if self.step_call_count % 100 == 0:
            print(f"✅ Hybrid wrapper step called {self.step_call_count} times")
        
        # Denormalize action from [-1, 1] to original action space
        action_original = self._denormalize_action(action)
        
        # APPLY SHIELD: Check and correct the denormalized action
        is_safe, corrected_action = self.shield_check(
            self._env.state if hasattr(self._env, 'state') else None,
            action_original,
            self._env
        )
        
        # Track shield interventions
        if not is_safe:
            self.shield_interventions += 1
        self.total_steps += 1
        
        # Step with the shielded action
        obs, reward, terminated, truncated, info = self._env.step(corrected_action)
        
        # Calculate cost from violations
        eq_viol = info.get('eq_viol', 0.0)
        ineq_viol = info.get('ineq_viol', 0.0)
        
        if isinstance(eq_viol, (np.ndarray, list)):
            eq_cost = float(np.sum(eq_viol))
        else:
            eq_cost = float(eq_viol)
            
        if isinstance(ineq_viol, (np.ndarray, list)):
            ineq_cost = float(np.sum(ineq_viol))
        else:
            ineq_cost = float(ineq_viol)
        
        cost_float = eq_cost + ineq_cost
        info['cost'] = cost_float
        info['shield_intervention'] = not is_safe
        info['shield_rate'] = self.shield_interventions / max(1, self.total_steps)
        
        # Track episode-level custom metrics
        self.ep_ineq_violations += ineq_cost
        if not is_safe:
            self.ep_shield_interventions += 1
        self.ep_steps += 1
        
        # At episode end, log custom metrics with special prefix for TensorBoard
        if terminated or truncated:
            info['Metrics/EpisodeIneqViolations'] = self.ep_ineq_violations
            info['Metrics/EpisodeShieldInterventions'] = self.ep_shield_interventions
            info['Metrics/ShieldInterventionRate'] = self.ep_shield_interventions / max(1, self.ep_steps)
            
            # Log directly to custom TensorBoard logger
            global _custom_logger
            if _custom_logger is not None:
                _custom_logger.log_step(info)
            
            # Reset episode counters
            self.ep_ineq_violations = 0.0
            self.ep_shield_interventions = 0
            self.ep_steps = 0
        
        # Convert to tensors
        observation = torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        cost = torch.tensor(cost_float, dtype=torch.float32)
        terminated = torch.tensor(bool(terminated), dtype=torch.bool)
        truncated = torch.tensor(bool(truncated), dtype=torch.bool)
        info['original_reward'] = reward.clone()
        
        return observation, reward, cost, terminated, truncated, info

def main():
    parser = argparse.ArgumentParser(description='Launch FOCOPS Hybrid (Shielded)')
    # You can now pass 'IEEE69-Hybrid-v0' here
    parser.add_argument('--env-id', type=str, default='IEEE33-Hybrid-v0', help='Environment ID (IEEE33-Hybrid-v0 or IEEE69-Hybrid-v0)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=500, help='Steps per epoch')
    parser.add_argument('--logdir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default 42 for Agent 3)')
    args = parser.parse_args()

    # Set default logdir based on env if not provided
    if args.logdir is None:
        args.logdir = f'logs/focops_hybrid_{args.env_id}'

    # Register 33-Bus Hybrid
    try:
        gym.spec('IEEE33-Hybrid-v0')
    except Exception:
        gym.register(id='IEEE33-Hybrid-v0', entry_point='__main__:IEEE33HybridWrapper')

    # Register 69-Bus Hybrid
    try:
        gym.spec('IEEE69-Hybrid-v0')
    except Exception:
        gym.register(id='IEEE69-Hybrid-v0', entry_point='__main__:IEEE69HybridWrapper')
    
    os.makedirs(args.logdir, exist_ok=True)

    config = {
        'seed': args.seed,
        'env_id': args.env_id,
        'exp_name': f'focops_hybrid_{args.env_id}', 
        'algo_cfgs': {
            'use_cost': True,
            'cost_gamma': 0.99,
            'clip_ratio': 0.2,
            'ent_coef': 0.0,
            'entropy_coef': 0.0,
            'value_coef': 0.5,
            'cost_value_coef': 0.5,
            'standardized_rew_adv': True,
            'standardized_cost_adv': True,
            'reward_normalize': False,
            'obs_normalize': False,
            'cost_normalize': False,
            'use_critic_norm': False,
            'critic_norm_coef': 0.001,
            'use_max_grad_norm': True,
            'max_grad_norm': 0.5,
            'adv_estimation_method': 'gae',
            'kl_early_stop': True,
            'update_iters': 5,
            'steps_per_epoch': args.steps_per_epoch,
            'batch_size': 256,
            'target_kl': 0.02,
            'gamma': 0.99,
            'lam': 0.95,
            'lam_c': 0.95,
            'cost_lam': 0.95,
            'focops_lam': 1.0,
            'penalty_coef': 1.0,
            'focops_eta': 0.02
        },
        'train_cfgs': {
            'device': 'cpu',
            'vector_env_nums': 1,
            'epochs': args.epochs,
            'steps_per_epoch': args.steps_per_epoch,
            'batch_size': 256,
            'cost_limit': 10.0,
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [64, 64],
                'activation': 'tanh',
                'lr': 3e-4,
            },
            'critic': {
                'hidden_sizes': [64, 64],
                'activation': 'tanh',
                'lr': 3e-4,
            },
            'weight_initialization_mode': 'orthogonal',
            'actor_type': 'gaussian_learning',
            'critic_type': 'mlp',
            'linear_lr_decay': False,
            'exploration_noise_anneal': False,
            'exploration_noise_init': 0.0,
            'exploration_noise_final': 0.0,
            'use_sde': False,
            'action_std': 0.1,
        },
        'logger_cfgs': {
            'log_dir': args.logdir,
            'use_wandb': False,
            'use_tensorboard': True,
            'save_model': True,
            'save_model_freq': 10,
        },
        'lagrange_cfgs': {
            'cost_limit': 10.0,
            'lagrangian_multiplier_init': 0.001,
            'lambda_lr': 0.01,
            'lambda_optimizer': 'Adam'
        }
    }
    
    cfgs = Config(**config)
    
    agent = FOCOPS(env_id=args.env_id, cfgs=cfgs)
    
    # After agent initialization, find OmniSafe's actual tensorboard directory
    global _custom_logger
    import glob
    import time
    
    # Wait a moment for OmniSafe to create directory structure
    time.sleep(2)
    
    # Find the MOST RECENT seed directory matching current seed
    seed_pattern = os.path.join(args.logdir, f'focops_hybrid_{args.env_id}', f'seed-{args.seed:03d}-*')
    seed_dirs = sorted(glob.glob(seed_pattern), key=os.path.getmtime, reverse=True)
    
    if seed_dirs:
        tb_log_dir = os.path.join(seed_dirs[0], 'tb')
        _custom_logger = CustomSafetyLogger(log_dir=tb_log_dir)
        print(f"\n📊 Custom Safety Logger writing to OmniSafe's TB dir: {tb_log_dir}")
        
        # Store logger globally so wrappers can access it
        import ieee33_wrapper, ieee69_wrapper
        ieee33_wrapper._custom_logger = _custom_logger
        ieee69_wrapper._custom_logger = _custom_logger
    else:
        print(f"⚠️  Warning: Could not find TensorBoard directory at {seed_pattern}")
    
    agent.learn()
    
    # Close custom logger
    if _custom_logger:
        _custom_logger.close()

if __name__ == '__main__':
    main()