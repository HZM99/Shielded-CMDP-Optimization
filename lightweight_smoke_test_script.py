# lightweight smoke-test script
#  ...existing code...
"""
Smoke test for OmniSafe agent initialization (no training).
- Validates that minimal configs allow creating agents for PPO-Lagrangian and FOCOPS.
- Runs quickly and prints clear diagnostics for missing keys / exceptions.
Run inside your omnisafe310 env:
    conda activate omnisafe310
    python scripts\smoke_test_omnisafe.py
"""
from pprint import pprint
import traceback

def safe_import(paths):
    for p in paths:
        try:
            mod = __import__(p, fromlist=['*'])
            return mod
        except Exception:
            continue
    return None

def build_minimal_cfg(overrides=None):
    """
    Build a Config-like dict with sensible minimal defaults OmniSafe expects.
    Use device 'cpu' and tiny steps to avoid heavy work.
    """
    base = {
        'seed': 0,
        'env_id': 'IEEE33-v0',
        'exp_name': 'smoke_test',
        'algo_cfgs': {
            'use_cost': True,
            'gamma': 0.99,
            'lam': 0.97,
            'lam_c': 0.97,
            'adv_estimation_method': 'gae',
            'standardized_rew_adv': True,
            'standardized_cost_adv': True,
            'reward_normalize': False,
            'cost_normalize': False,
            'obs_normalize': False,
            'steps_per_epoch': 10,
            'update_iters': 1,
            'batch_size': 32,
            'target_kl': 0.02,
            'kl_early_stop': False,
            # FOCOPS-specific placeholders
            'penalty_coef': 1.0,
            'focops_lam': 1.0,
            'focops_eta': 0.02,
        },
        'train_cfgs': {
            'device': 'cpu',
            'vector_env_nums': 1,
            'epochs': 1,
            'steps_per_epoch': 10,
            'batch_size': 32,
            'cost_limit': 1.0,
            'log_interval': 10,
            'eval_interval': 10,
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
            'actor_type': 'gaussian_learning',
            'critic_type': 'mlp',
            'weight_initialization_mode': 'xavier_uniform',
            'linear_lr_decay': False,
            # OmniSafe PolicyGradient checks this flag; set explicit default
            'exploration_noise_anneal': False,
            # Safe no-op defaults for exploration parameters
            'exploration_noise': 0.0,
        },
        # Provide minimal keys required by standard Lagrange used in FOCOPS.
        # We intentionally avoid PID-specific keys and will filter by module during discovery.
        'lagrange_cfgs': {
            'cost_limit': 1.0,
            'lagrangian_multiplier_init': 0.1,
            'lambda_lr': 1e-3,
            'lambda_optimizer': 'Adam',
        },
        'logger_cfgs': {
            'log_dir': 'train/smoke_test_logs',
            'use_wandb': False,
            'use_tensorboard': False,
            'save_model': False,
            'save_model_freq': 0,
        }
    }
    if overrides:
        # shallow merge for convenience (sufficient for smoke test)
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k].update(v)
            else:
                base[k] = v
    return base

def _discover_algo_class(algo_name):
    """Search omnisafe.algorithms for a class matching algo_name hints.
    Returns the class object or None.
    """
    try:
        import importlib, pkgutil, inspect
        root = importlib.import_module('omnisafe.algorithms')
        name_lower = algo_name.lower()
        if 'ppo' in name_lower:
            hints = ['ppo', 'lag']  # match lag/lagr/lagrangian; avoid pid_* variants
        elif 'focops' in name_lower:
            hints = ['focops']
        else:
            hints = [name_lower]
        for _, modname, _ in pkgutil.walk_packages(root.__path__, root.__name__ + '.'):
            if 'on_policy' not in modname:
                continue
            try:
                m = importlib.import_module(modname)
            except Exception:
                continue
            for _, obj in inspect.getmembers(m, inspect.isclass):
                qual = f"{obj.__module__}.{obj.__name__}".lower()
                # Exclude PID Lagrange variants when searching for PPO-Lagrangian
                if 'ppo' in name_lower and 'pid' in qual:
                    continue
                if all(h in qual for h in hints):
                    return obj
    except Exception:
        return None
    return None

def try_instantiate(algo_name, import_paths, cfg_dict):
    print(f"\n=== Testing {algo_name} ===")
    mod = safe_import(import_paths)
    cls = None
    if mod is None:
        print(f"Info: direct import failed for {algo_name}. Falling back to dynamic scanning.")
        cls = _discover_algo_class(algo_name)
    else:
        # try common class names within the imported module
        for name in ('FOCOPS', 'FOCOPSAlgo', 'PPOLagrangian', 'PPOLag', 'PPOLagr'):
            cls = getattr(mod, name, None)
            if cls:
                break
        if cls is None:
            cls = _discover_algo_class(algo_name)
    if cls is None:
        print(f"FAIL: Could not import or discover class for {algo_name}. Tried modules:\n  " + "\n  ".join(import_paths))
        return False
    print(f"Using class: {cls.__module__}.{cls.__name__}")

    try:
        from omnisafe.utils.config import Config
        cfg = Config(**cfg_dict)
    except Exception as e:
        print("FAIL: could not build Config object from dict:")
        traceback.print_exc()
        return False

    try:
        # Instantiate agent (do not call learn())
        agent = cls(env_id=cfg.env_id, cfgs=cfg)
        print("PASS: Agent instantiated successfully (init completed).")
        # try calling a light internal init method if available (quiet)
        if hasattr(agent, '_init'):
            try:
                agent._init()
            except Exception:
                # ignore; instantiation already called init in many implementations
                pass
        return True
    except Exception:
        print("FAIL: Exception during agent instantiation:")
        traceback.print_exc()
        return False

def main():
    print("OmniSafe smoke test (instantiate agents only).")
    cfg = build_minimal_cfg()

    # import local wrapper so environment id is registered
    try:
        import ieee33_wrapper  # noqa: F401
        print("Imported local IEEE33 wrapper.")
    except Exception as e:
        print("Warning: could not import local ieee33_wrapper:", e)

    # Test FOCOPS
    focops_paths = [
        'omnisafe.algorithms.on_policy.first_order.focops',
        'omnisafe.algorithms.on_policy.first_order.FOCOPS',
        'omnisafe.algorithms.on_policy.focops',
    ]
    focops_ok = try_instantiate('FOCOPS', focops_paths, cfg)

    # Test PPO-Lagrangian (try several plausible paths)
    ppo_paths = [
        'omnisafe.algorithms.on_policy.ppo_lagrangian',
        'omnisafe.algorithms.on_policy.first_order.ppo_lagrangian',
        'omnisafe.algorithms.on_policy.ppo_lagrangian.ppo_lagrangian',
        'omnisafe.algorithms.on_policy.ppo_lagrangian.PPOLagrangian',
        'omnisafe.algorithms.on_policy.ppo_lagrangian.ppo',
    ]
    ppo_ok = try_instantiate('PPO-Lagrangian', ppo_paths, cfg)

    print("\nSummary:")
    print(f" FOCOPS: {'OK' if focops_ok else 'FAIL'}")
    print(f" PPO-Lagrangian: {'OK' if ppo_ok else 'FAIL'}")

if __name__ == '__main__':
    main()
# ...existing code...