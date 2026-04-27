import os
import argparse
import gymnasium as gym
from omnisafe.algorithms.on_policy.first_order.focops import FOCOPS
from omnisafe.utils.config import Config
from custom_logger import CustomSafetyLogger

# Global custom logger instance
_custom_logger = None

def main():
    parser = argparse.ArgumentParser(description='Launch FOCOPS on IEEE33 env')
    parser.add_argument('--env-id', type=str, default='IEEE33-v0', help='Environment ID (IEEE33-v0 or OmniIEEE33-v0)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=500, help='Steps per epoch')
    parser.add_argument('--logdir', type=str, default='logs/focops_agent2_run', help='TensorBoard log directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    try:
        gym.spec('IEEE33-v0')
    except Exception:
        gym.register(
            id='IEEE33-v0',
            entry_point='ieee33_wrapper:IEEE33Wrapper'
        )
    try:
        import ieee33_wrapper
        import ieee69_wrapper
    except Exception as e:
        print(f'Warning: failed to import local wrappers: {e}')
    
    # Ensure log directory exists for TensorBoard
    os.makedirs(args.logdir, exist_ok=True)

    config = {
        'seed': args.seed,
        'env_id': args.env_id,
        'exp_name': 'focops_agent2_run',
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
        }
        ,
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
    # OmniSafe creates directory based on current seed
    global _custom_logger
    import glob
    import time
    
    # Wait a moment for OmniSafe to create directory structure
    time.sleep(2)
    
    # Find the MOST RECENT seed directory (the one just created)
    seed_pattern = os.path.join(args.logdir, 'focops_agent2_run', f'seed-{args.seed:03d}-*')
    seed_dirs = sorted(glob.glob(seed_pattern), key=os.path.getmtime, reverse=True)
    
    if seed_dirs:
        tb_log_dir = os.path.join(seed_dirs[0], 'tb')
        _custom_logger = CustomSafetyLogger(log_dir=tb_log_dir)
        print(f"\n📊 Custom Safety Logger writing to OmniSafe's TB dir: {tb_log_dir}")
        
        # Store logger globally so wrapper can access it
        import ieee69_wrapper
        ieee69_wrapper._custom_logger = _custom_logger
    else:
        print(f"⚠️  Warning: Could not find TensorBoard directory at {seed_pattern}")
    
    agent.learn()
    
    # Close custom logger
    if _custom_logger:
        _custom_logger.close()

if __name__ == '__main__':
    main()