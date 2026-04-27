import os
import sys
import gymnasium as gym
import yaml
from omnisafe.algorithms import PolicyGradient
from omnisafe.utils.config import Config

# Helper: recursive dict update (merge src into dst)
def recursive_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            recursive_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def create_default_config():
    """Create default configuration for PPO-Lagrangian"""
    return {
        'defaults': {
            'env_id': 'IEEE33-v0',  # Our environment ID
            'algo_cfgs': {
                'use_cost': True,
                'cost_gamma': 0.99,
                'cost_normalize': False,
                'clip_ratio': 0.2,
                'ent_coef': 0.0,
                'reward_normalize': False,
                'obs_normalize': False,
                'value_coef': 0.5,
                'cost_value_coef': 0.5,
                'standardized_rew_adv': True,
                'standardized_cost_adv': True,
                'penalty_coef': 1.0,
                'use_max_grad_norm': True,
                'max_grad_norm': 0.5,
                'use_critic_norm': True,
                'critic_norm_coef': 0.001,
                'adv_estimation_method': 'gae',  # Required by OmniSafe
                'kl_early_stop': True  # Required for PPO update
            },
            'train_cfgs': {
                'device': 'cpu',
                'vector_env_nums': 1,
                'epochs': 50,
                'steps_per_epoch': 1000,
                'update_iters': 5,
                'batch_size': 256,
                'target_kl': 0.02,
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
                # OmniSafe expects some model-level defaults
                'weight_initialization_mode': 'orthogonal',
                'actor_type': 'gaussian_learning',  # Support continuous action spaces with learned std
                'critic_type': 'mlp',
                'linear_lr_decay': False,
                # Additional model defaults OmniSafe may expect
                'exploration_noise_anneal': False,
                'exploration_noise_init': 0.0,
                'exploration_noise_final': 0.0,
                'use_sde': False,
                'action_std': 0.1,
            },
            'logger_cfgs': {
                'log_dir': 'train/PPO_Lagrangian_logs',
                'use_wandb': False,
                'use_tensorboard': True,
                'save_model': True,
                'save_model_freq': 10,  # Save model every 10 epochs
                'log_interval': 1000,
                'eval_interval': 10000
            },
            'lagrange_cfgs': {
                'lagrangian_multiplier_init': 0.1,
                'lambda_optimizer': 'Adam',
                'lambda_lr': 1e-3,
                'lambda_clip': False
            }
        }
    }

def main():
    # Register environment if not already registered
    try:
        gym.spec('IEEE33-v0')
    except Exception:
        gym.register(
            id='IEEE33-v0',
            entry_point='ieee33_wrapper:IEEE33Wrapper'
        )
    # Import local wrapper to ensure it registers with OmniSafe's registry
    try:
        import ieee33_wrapper  # noqa: F401 - triggers env_register decorator
    except Exception as e:
        print(f'Warning: failed to import local ieee33_wrapper: {e}')
    
    # Config file path
    default_cfg_path = os.path.join(os.getcwd(), 'ppo_lagrangian_cfg.yaml')
    
    # Get default config
    DEFAULT_CONFIG = create_default_config()

    # Create or load config file
    if not os.path.exists(default_cfg_path):
        with open(default_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, indent=4)
    
    # Load config
    try:
        with open(default_cfg_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        config_data = DEFAULT_CONFIG

    # Ensure we have valid data
    if config_data is None:
        config_data = DEFAULT_CONFIG
    
    # The YAML uses a top-level 'defaults' section
    defaults = config_data.get('defaults', config_data)

    # Minimal overrides you want to apply to the default config
    overrides = {
        'env_id': 'IEEE33-v0',
        'log_dir': os.path.join(os.getcwd(), 'train', 'PPOLagrangian_logs'),
        'seed': 0,
        'exp_name': 'PPOLagrangian_IEEE33Wrapper',
    }

    # Merge overrides into defaults and create Config
    merged = recursive_update(defaults.copy(), overrides)
    cfgs = Config(**merged)

    # Ensure logger_cfgs exists
    if not hasattr(cfgs, 'logger_cfgs'):
        logger_defaults = {
            'log_dir': os.path.join(os.getcwd(), 'train', 'PPOLagrangian_logs'),
            'use_wandb': False,
            'use_tensorboard': True,
            'save_model': True,
            'save_model_freq': 10,  # Save model every 10 epochs
            'log_interval': 1000,
            'eval_interval': 10000
        }
        cfgs.logger_cfgs = Config(**logger_defaults)

    # Ensure algo_cfgs exists and has sensible defaults
    if not hasattr(cfgs, 'algo_cfgs'):
        algo_defaults = {
            'use_cost': True,
            'cost_gamma': 0.99,
            'gamma': 0.99,
            'lam': 0.95,
            'cost_normalize': False,
            'clip_ratio': 0.2,
            'ent_coef': 0.0,
            'reward_normalize': False,
            'obs_normalize': False,
            'value_coef': 0.5,
            'cost_value_coef': 0.5,
            'standardized_rew_adv': True,
            'standardized_cost_adv': True,
            'penalty_coef': 1.0,
            'use_max_grad_norm': True,
            'max_grad_norm': 0.5,
            'use_critic_norm': True,
            'critic_norm_coef': 0.001,
            'adv_estimation_method': 'gae',  # Required by OmniSafe
            'kl_early_stop': True  # Required for PPO update
        }
        cfgs.algo_cfgs = Config(**algo_defaults)

    # Mirror important training keys OmniSafe expects into algo_cfgs
    try:
        train = cfgs.train_cfgs
    except Exception:
        train = None

    if train is not None:
        for key in ['steps_per_epoch', 'batch_size', 'update_iters', 'target_kl', 'epochs', 'vector_env_nums']:
            # If algo_cfgs doesn't have the key, copy from train_cfgs
            if not hasattr(cfgs.algo_cfgs, key) and hasattr(train, key):
                try:
                    setattr(cfgs.algo_cfgs, key, getattr(train, key))
                except Exception:
                    # best-effort, ignore if cannot set
                    pass

    # Ensure lagrangian configs
    if not hasattr(cfgs, 'lagrange_cfgs'):
        lagrange_defaults = {
            'cost_limit': cfgs.train_cfgs.cost_limit,
            'lagrangian_multiplier_init': 0.001,
            'lambda_lr': 0.01,
            'lambda_optimizer': 'Adam'
        }
        cfgs.lagrange_cfgs = Config(**lagrange_defaults)

    # Ensure core RL keys exist both at algo_cfgs and top-level if needed
    try:
        if not hasattr(cfgs.algo_cfgs, 'gamma'):
            cfgs.algo_cfgs.gamma = 0.99
        if not hasattr(cfgs.algo_cfgs, 'lam'):
            cfgs.algo_cfgs.lam = 0.95
        if not hasattr(cfgs.algo_cfgs, 'lam_c'):
            cfgs.algo_cfgs.lam_c = 0.95
    except Exception:
        # If algo_cfgs not present or not a Config, set top-level aliases
        setattr(cfgs, 'gamma', 0.99)
        setattr(cfgs, 'lam', 0.95)
    
    # Mirror to top-level for any code expecting cfgs.gamma
    if not hasattr(cfgs, 'gamma'):
        setattr(cfgs, 'gamma', cfgs.algo_cfgs.gamma if hasattr(cfgs, 'algo_cfgs') and hasattr(cfgs.algo_cfgs, 'gamma') else 0.99)
    if not hasattr(cfgs, 'lam'):
        setattr(cfgs, 'lam', cfgs.algo_cfgs.lam if hasattr(cfgs, 'algo_cfgs') and hasattr(cfgs.algo_cfgs, 'lam') else 0.95)

    # Create and train agent
    agent = PolicyGradient(cfgs.env_id, cfgs=cfgs)  # Pass the environment ID from config
    agent.learn()

if __name__ == '__main__':
    main()