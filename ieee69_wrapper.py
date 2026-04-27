import gymnasium as gym
import torch
import torch.nn as nn
from rl_constrained_smartgrid_control.environments.bus69_environment import IEEE69BusEnv
from omnisafe.envs.core import CMDP, env_register
import numpy as np

# Module-level custom logger (set by launch script)
_custom_logger = None

@env_register
class IEEE69Wrapper(CMDP):
    """Wrapper that adapts the IEEE69BusEnv to use normalized action space compatible with OmniSafe."""

    # Support both the canonical ID and an alias often used in configs
    _support_envs = ['IEEE69-v0', 'OmniIEEE69-v0']
    need_time_limit_wrapper = True
    need_auto_reset_wrapper = True

    def __init__(self, env_id: str, num_envs: int = 1, device: str = 'cpu', **kwargs):
        super().__init__(env_id)
        # Initialize base environment
        self._env = IEEE69BusEnv()
        self._num_envs = num_envs

        # Store original action space
        self.original_action_space = self._env.action_space

        # Create normalized action space in [-1, 1]
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._env.action_space.shape,
            dtype=np.float32
        )
        
        # Set observation space from env
        self._observation_space = self._env.observation_space
        
        # Set metadata (required by OmniSafe)
        self._metadata = {'render_modes': None}
        
        # Episode-level tracking for custom metrics
        self.ep_ineq_violations = 0.0
        self.ep_steps = 0

    def step(self, action):
        # Denormalize action from [-1, 1] to original action space
        action_original = self._denormalize_action(action)
        
        # Step in original environment
        obs, reward, terminated, truncated, info = self._env.step(action_original)

        # Get the actual violation arrays from the info dict
        eq_viol = info.get('eq_viol', 0.0)
        ineq_viol = info.get('ineq_viol', 0.0)

        # Convert to numpy if needed and sum to get scalar values
        if isinstance(eq_viol, (np.ndarray, list)):
            eq_cost = float(np.sum(eq_viol))
        else:
            eq_cost = float(eq_viol)
            
        if isinstance(ineq_viol, (np.ndarray, list)):
            ineq_cost = float(np.sum(ineq_viol))
        else:
            ineq_cost = float(ineq_viol)

        # Total cost as a single float
        cost_float = eq_cost + ineq_cost

        # Add this real cost to the info dict for logging/callbacks
        info['cost'] = cost_float
        
        # Track episode-level custom metrics
        self.ep_ineq_violations += ineq_cost
        self.ep_steps += 1
        
        # At episode end, log custom metrics with special prefix for TensorBoard
        if terminated or truncated:
            info['Metrics/EpisodeIneqViolations'] = self.ep_ineq_violations
            info['Metrics/EpisodeShieldInterventions'] = 0  # No shield in Agent 2
            info['Metrics/ShieldInterventionRate'] = 0.0  # No shield in Agent 2
            
            # Log directly to custom TensorBoard logger if available
            global _custom_logger
            if _custom_logger is not None:
                _custom_logger.log_step(info)
            
            # Reset episode counters
            self.ep_ineq_violations = 0.0
            self.ep_steps = 0

        # Convert to tensors with consistent shapes
        observation = torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        cost = torch.tensor(cost_float, dtype=torch.float32)

        # Convert to proper bool tensors
        terminated = torch.tensor(bool(terminated), dtype=torch.bool)
        truncated = torch.tensor(bool(truncated), dtype=torch.bool)

        # Store original reward as tensor for logging
        info['original_reward'] = reward.clone()

        return observation, reward, cost, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def set_seed(self, seed):
        """Set environment seed.

        This sets numpy and torch seeds and attempts to set the underlying
        environment/data seeds if available.
        """
        if seed is None:
            return
        try:
            import numpy as _np

            _np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch as _torch

            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        # Try to set seed on underlying components if they expose such API
        try:
            if hasattr(self._env, 'set_seed'):
                self._env.set_seed(seed)
            elif hasattr(self._env, 'data') and hasattr(self._env.data, 'set_seed'):
                self._env.data.set_seed(seed)
        except Exception:
            # Non-critical; continue without failing
            pass

    def render(self):
        """Render environment."""
        return self._env.render()

    def close(self):
        """Close environment."""
        self._env.close()
        
    def _normalize_action(self, action):
        """Convert from original action space to [-1, 1]."""
        # Convert to numpy if it's a torch tensor
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
            
        low = self.original_action_space.low
        high = self.original_action_space.high
            
        # Keep original shape
        action_normalized = 2.0 * ((action - low) / (high - low)) - 1.0
        return action_normalized

    def _denormalize_action(self, normalized_action):
        """Convert from [-1, 1] to original action space."""
        # Convert to numpy if it's a torch tensor
        if isinstance(normalized_action, torch.Tensor):
            normalized_action = normalized_action.detach().cpu().numpy()
            
        low = self.original_action_space.low
        high = self.original_action_space.high
        
        # Handle action shape properly - keep the original shape
        action = low + (normalized_action + 1.0) * 0.5 * (high - low)
        # Return flattened array since that's what the env expects
        return np.clip(action, low, high).flatten()

    @property
    def max_episode_steps(self):
        """Maximum episode steps."""
        return 1000  # Adjust this based on your environment
