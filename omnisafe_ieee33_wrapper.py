from omnisafe.envs.base import BaseEnv
from rl_constrained_smartgrid_control.environments.bus33_environment import IEEE33BusEnv

class OmniSafeIEEE33Wrapper(BaseEnv):
    def __init__(self):
        super().__init__()
        self.env = IEEE33BusEnv()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get("ineq_viol", 0) + info.get("eq_viol", 0)
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()