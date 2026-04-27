import gymnasium as gym
from gymnasium.envs.registration import register

# Import local wrappers to trigger OmniSafe's env_register decorator
try:
    import ieee33_wrapper  # noqa: F401
except Exception:
    # Non-fatal: direct Gym registration below still allows gym.make
    pass

try:
    import ieee69_wrapper  # noqa: F401
except Exception:
    pass

# Direct Gym registrations for raw bus envs
gym.register(
    id='IEEE33Bus-v0',
    entry_point='rl_constrained_smartgrid_control.environments.bus33_environment:IEEE33BusEnv',
)

gym.register(
    id='IEEE69Bus-v0',
    entry_point='rl_constrained_smartgrid_control.environments.bus69_environment:IEEE69BusEnv',
)

# Optional: register an alias that points to the OmniSafe-compatible wrapper
# This lets gym.make('OmniIEEE33-v0') work in scripts that expect that ID.
register(
    id='OmniIEEE33-v0',
    entry_point='ieee33_wrapper:IEEE33Wrapper',
)

register(
    id='OmniIEEE69-v0',
    entry_point='ieee69_wrapper:IEEE69Wrapper',
)