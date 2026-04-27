from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class SmartGridEnvironment(gym.Env, ABC):
    """
    Base class for smart grid environments.
    Implements the Constrained Markov Decision Process (CMDP) interface.
    All specific smart grid environments should inherit from this class and
    implement the abstract methods.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.max_steps = 1000  # Default max steps per episode

    @abstractmethod
    def step(
            self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.

        Parameters:
            action (np.ndarray): The action taken by the agent.

        Returns:
            observation (np.ndarray): The next observation of the environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict): Additional information about the step.
        """
        pass

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the state of the environment to an initial state.

        Parameters:
            seed (Optional[int]): Seed for random number generator.
            options (Optional[Dict]): Additional options for resetting.

        Returns:
            observation (np.ndarray): The initial observation of the environment.
            info (Dict): Additional information about the reset.
        """
        pass
