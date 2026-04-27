import os
import sys
import unittest

import gymnasium as gym
from omnisafe.algorithms import FOCOPS
from omnisafe.common.logger import Logger

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from launch_focops import get_config
from ieee33_wrapper import IEEE33Wrapper

class TestFOCOPSSmoke(unittest.TestCase):
    def setUp(self):
        """Set up test environment and configuration."""
        self.config = get_config()
        
    def test_environment_creation(self):
        """Test that the IEEE33 environment can be created."""
        env = gym.make('IEEE33-v0')
        wrapped_env = IEEE33Wrapper(env)
        self.assertIsNotNone(wrapped_env)
        
    def test_agent_initialization(self):
        """Test that FOCOPS agent can be initialized with config."""
        logger = Logger(
            log_dir='./logs/test',
            exp_name='focops_smoke_test',
            seed=1,
            use_tensorboard=False,
            use_wandb=False
        )
        
        agent = FOCOPS(
            env_id='IEEE33-v0',
            logger=logger,
            cfgs=self.config
        )
        self.assertIsNotNone(agent)

if __name__ == '__main__':
    unittest.main()