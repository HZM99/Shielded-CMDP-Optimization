"""
Electric Vehicle (EV) specific utilities for smart grid control environments.

This module defines the `ElectricVehicle` class, which extends the `Battery` class
to include EV-specific functionality, such as handling availability data and
adjusting actions based on operational constraints.
"""
from typing import Tuple

import numpy as np

from rl_constrained_smartgrid_control.environments.utils.battery import Battery


class ElectricVehicle(Battery):
    """
    Electric Vehicle (EV) class for managing EV-specific behavior in smart grid control.

    This class extends the `Battery` class to include additional functionality for
    handling EV availability data and adjusting actions based on EV-specific constraints.

    Attributes:
        availability_data (Any): Data representing the availability of the EV.
        availability_mask (Optional[np.ndarray]): Mask indicating the availability of the EV.
    """

    def __init__(
        self,
        power_data: np.ndarray,
        availability_data: np.ndarray,
        num: int,
        low: float = 10,
        high: float = 80,
        p_min: float = -20,
        p_max: float = 20,
        eta_in: float = 0.9,
        eta_out: float = 0.9,
        init_strategy: str = "full",
    ):
        """
        Initialize the ElectricVehicle instance.

        Args:
            power_data (np.ndarray): Power data for the EV. Shape: (T,).
            availability_data (np.ndarray): Availability data for the EV. Shape: (T,).
            num (int): Number of EVs.
            low (float): Minimum state of charge (SOC) as a percentage of capacity.
            high (float): Maximum state of charge (SOC) as a percentage of capacity.
            p_min (float): Minimum power output (discharge) in kW.
            p_max (float): Maximum power input (charge) in kW.
            eta_in (float): Charging efficiency (0 < eta_in <= 1).
            eta_out (float): Discharging efficiency (0 < eta_out <= 1).
            init_strategy (str): Strategy for initializing SOC. Options: "full", "empty", "random".

        Raises:
            ValueError: If `init_strategy` is not one of "full", "empty", or "random".
        """
        super(
            ElectricVehicle,
            self).__init__(
            power_data,
            num,
            low,
            high,
            p_min,
            p_max,
            eta_in,
            eta_out,
            init_strategy)
        self.availability_data = availability_data
        self.availability_mask = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single step in the EV simulation.

        This method updates the EV's state based on the given action, clips the action
        to respect operational constraints, and calculates the inequality distance.

        Args:
            action (np.ndarray): The action to be applied to the EV. Shape: (num,).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Updated state of the EV. Shape: (num,).
                - Inequality distance for the action. Shape: (num,).
        """
        upper_bound = np.ones((self.num, 1)) * self.high
        updated_high = np.concatenate(
            (self.p_max[:, None], (upper_bound - self.state)[:, None]), axis=1)
        updated_low = np.concatenate(
            (self.p_min[:, None], (upper_bound - self.state)[:, None]), axis=1)

        updated_high = np.min(updated_high, axis=1)
        updated_low = np.max(updated_low, axis=1)

        clipped_action = np.clip(action, updated_low, updated_high)
        self.state += clipped_action

        return self.state, self.ineq_dist(action, updated_low, updated_high)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the EV's state and availability mask.

        This method resets the EV's state of charge (SOC) based on the initialization
        strategy and fetches the availability mask from the availability data.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Reset state of the EV. Shape: (num,).
                - Availability mask for the EV. Shape: (T,).

        Raises:
            NotImplementedError: If the initialization strategy is not implemented.
        """
        self.availability_data.reset()
        self.p_data.reset()
        if self.init_strategy == "random":
            self.state = np.random.rand(
                self.num) * (self.high - self.low) + self.low
        elif self.init_strategy == "empty":
            self.state = np.ones(self.num) * self.low
        elif self.init_strategy == "full":
            self.state = np.ones(self.num) * self.high
        else:
            raise NotImplementedError(
                f"Initialization strategy {self.init_strategy} not implemented.")

        self.availability_mask = self.availability_data.fetch()

        return self.state.copy(), self.availability_mask.copy()
