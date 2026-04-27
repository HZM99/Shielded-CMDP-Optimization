# Save this file as: shield_wrapper.py

import gymnasium as gym  # <-- CORRECTED
import numpy as np
from shield_model import check_action_safety  # This file is still correct

class SafetyShieldWrapper(gym.Wrapper):  # <-- CORRECTED
    """
    A Gymnasium Wrapper that implements a Safety Shield (Action Shielding).
    
    It intercepts actions from the RL agent *before* they
    are applied to the real environment. It uses our custom
    'check_action_safety' function to validate and correct them.
    """
    
    def __init__(self, env):
        # Initialize the parent gym.Wrapper
        super().__init__(env)
        
        # We must store the last observation for our safety check
        self.last_observation = None
        print("✅ SafetyShieldWrapper (Gymnasium) initialized.")
        print("Agent actions will now be checked and corrected.")

    def reset(self, **kwargs):
        """
        Resets the environment and stores the initial observation.
        Your env's reset returns (obs, info).
        """
        # Call the base environment's reset()
        reset_result = self.env.reset(**kwargs)
        obs, info = reset_result  # Unpack 2-tuple
        
        self.last_observation = obs
        return obs, info

    def step(self, action):
        """
        The core of the Safety Shield.
        
        1. Intercepts the agent's 'action'.
        2. Checks it using 'check_action_safety'.
        3. Applies the (potentially corrected) 'safe_action' to the env.
        """
        
        # 1. Check the proposed action for safety
        is_safe, corrected_action = check_action_safety(
            self.last_observation, 
            action, 
            self.env  # Pass the base env so the shield can read its limits
        )
        
        # 2. Apply the *corrected* (and now safe) action to the real env
        # Your env's step returns (obs, reward, done, truncated, info)
        obs, reward, done, truncated, info = self.env.step(corrected_action)
        
        # 3. Store the new observation for the next safety check
        self.last_observation = obs
        
        # 4. Add information for your research/logging
        # This lets you log how often the shield had to intervene.
        info['shield_intervention'] = not is_safe
        info['original_action'] = action
        
        return obs, reward, done, truncated, info