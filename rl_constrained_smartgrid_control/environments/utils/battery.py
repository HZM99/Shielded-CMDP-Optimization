import logging
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch


class InitStrategy(Enum):
    FULL = "full"
    EMPTY = "empty"
    RANDOM = "random"


class Battery:
    DEFAULT_LOW_SOC = 0.1
    DEFAULT_HIGH_SOC = 0.8
    DEFAULT_P_MIN = -0.2
    DEFAULT_P_MAX = 0.2
    DEFAULT_ETA_IN = 0.9
    DEFAULT_ETA_OUT = 0.9
    DEFAULT_INIT_STRATEGY = InitStrategy.FULL
    DEFAULT_RESIDUAL = 1.0

    def __init__(
        self,
        p_data: np.ndarray,
        num: int,
        genbase: float,
        low: float = DEFAULT_LOW_SOC,
        high: float = DEFAULT_HIGH_SOC,
        p_min: float = DEFAULT_P_MIN,
        p_max: float = DEFAULT_P_MAX,
        eta_in: float = DEFAULT_ETA_IN,
        eta_out: float = DEFAULT_ETA_OUT,
        init_strategy: InitStrategy = DEFAULT_INIT_STRATEGY,
        residual: float = DEFAULT_RESIDUAL,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Battery with given parameters.

        Args:
            p_data (np.ndarray): Power data for the battery. Shape: (T,).
            num (int): Number of batteries.
            genbase (float): Base generation value.
            low (float): Minimum state of charge (SOC) as a fraction of capacity.
            high (float): Maximum state of charge (SOC) as a fraction of capacity.
            p_min (float): Minimum power output (discharge) in per unit.
            p_max (float): Maximum power input (charge) in per unit.
            eta_in (float): Charging efficiency.
            eta_out (float): Discharging efficiency.
            init_strategy (InitStrategy): Strategy for initializing SOC.
            residual (float): Residual SOC after discharge in per unit.
            device (torch.device): Device to run computations on (CPU or GPU).
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Battery...")

        if isinstance(init_strategy, str):
            try:
                init_strategy = InitStrategy(init_strategy)
            except ValueError:
                raise ValueError(
                    f"Invalid initialization strategy: {init_strategy}")

        self._validate_inputs(
            low,
            high,
            p_min,
            p_max,
            eta_in,
            eta_out,
            init_strategy)

        self.p_data = p_data
        self.num = num
        self.genbase = genbase
        self.low = low
        self.high = high
        self.p_min = p_min
        self.p_max = p_max
        self.eta_in = eta_in
        self.eta_out = eta_out
        self.init_strategy = init_strategy
        self.residual = residual
        self.state = None
        self.device = device

        self.logger.info(
            f"Battery initialized with {num} units and generation base {genbase}.")

    def _validate_inputs(
        self, low: float, high: float, p_min: float, p_max: float, eta_in: float, eta_out: float, init_strategy: str
    ) -> None:
        """Validate input parameters."""
        if isinstance(init_strategy, InitStrategy):
            init_strategy = init_strategy.value

        if not (0 <= low <= 1):
            raise ValueError("low must be between 0 and 1.")
        if not (0 <= high <= 1):
            raise ValueError("high must be between 0 and 1.")
        if not (low <= high):
            raise ValueError("low must be less than or equal to high.")
        if not (p_min <= 0):
            raise ValueError("p_min must be less than or equal to 0.")
        if not (p_max >= 0):
            raise ValueError("p_max must be greater than or equal to 0.")
        if not (0 <= eta_in <= 1):
            raise ValueError("eta_in must be between 0 and 1.")
        if not (0 <= eta_out <= 1):
            raise ValueError("eta_out must be between 0 and 1.")
        if init_strategy not in [strategy.value for strategy in InitStrategy]:
            raise ValueError(
                f"init_strategy must be one of {[e.value for e in InitStrategy]}.")

    def initialize_state(self) -> None:
        """Initialize the battery state based on the chosen strategy."""
        strategies = {
            InitStrategy.FULL: lambda: np.ones(self.num) * self.high,
            InitStrategy.EMPTY: lambda: np.ones(self.num) * (self.low + 0.1),
            InitStrategy.RANDOM: lambda: np.random.dirichlet(np.ones(self.num)) * (self.residual + self.low),
        }
        self.state_battery = strategies[self.init_strategy]()
        self.logger.info(
            f"Battery state initialized using strategy: {self.init_strategy.value}")
        return self.state_battery

    def reset(self) -> Optional[np.ndarray]:
        """
        Reset the battery state to its initial configuration.

        This method resets the battery state by:
            1. Normalizing the power data (`p_data`) based on the generation base (`genbase`).
            2. Initializing the battery state (`state_battery`) using the chosen strategy
               (e.g., FULL, EMPTY, RANDOM).
            3. Concatenating the normalized power data and the initialized battery state
               to form the complete state.

        Returns:
            Optional[np.ndarray]: The reset battery state as a NumPy array, or None if an
            error occurs during initialization.


        Raises:
            NotImplementedError: If there is an error during battery state initialization.
        """
        power_state, _ = self.p_data.reset()
        power_state /= self.genbase

        try:
            battery_state = self.initialize_state()
        except Exception as e:
            raise NotImplementedError(f"Error initializing battery state: {e}")

        self.state = np.concatenate([power_state, battery_state])
        self.logger.info("Battery state has been reset.")

        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Perform a single step in the battery simulation.

        This method updates the battery state based on the given action, calculates the
        associated cost, and returns the updated state and cost.

        Steps:
            1. Fetch the current power state (`power_state`) and normalize it based on the generation base.
            2. Clip the action to ensure it respects the battery's operational constraints.
            3. Update the battery state (`battery_state`) based on the clipped action.
            4. Calculate the cost associated with the action.
            5. Return the total cost and the updated state.

        Args:
            action (np.ndarray): The action to be applied to the battery. Shape: (num,).

        Returns:
            Tuple[float, np.ndarray]:
                - The total cost incurred by the action.
                - The updated battery state as a NumPy array.
        """
        power_state, _ = self.p_data.fetch()
        power_state /= self.genbase

        battery_state = self.state[: self.num]
        price = self.state[self.num]

        upper_bound = np.ones(self.num) * self.high
        lower_bound = np.ones(self.num) * self.low

        max_power = (
            1
            / self.eta_in
            * np.concatenate(
                (self.p_max * np.ones((self.num, 1)),
                 (upper_bound - battery_state)[:, None]),
                axis=1,
            )
        )
        min_power = self.eta_out * np.concatenate(
            (self.p_min * np.ones((self.num, 1)),
             (lower_bound - battery_state)[:, None]),
            axis=1,
        )

        max_power = np.min(max_power, axis=1)
        min_power = np.max(min_power, axis=1)

        clipped_action = np.clip(action, min_power, max_power)
        processed_action = self.process_action(clipped_action)

        battery_state += processed_action
        self.state = np.concatenate([battery_state, power_state])

        cost = -processed_action * price
        total_cost = cost.sum()

        return total_cost, self.state.copy()

    def ineq_resid(self, state: torch.Tensor,
                   action: torch.Tensor) -> torch.Tensor:
        """
        Compute the inequality residuals for the given state and action.

        This method calculates the residuals for the inequality constraints of the battery
        system, ensuring that the action respects the operational bounds.

        Args:
            state (torch.Tensor): The current state of the battery. Shape: (batch_size, num).
            action (torch.Tensor): The action to be applied to the battery. Shape: (batch_size, num).

        Returns:
            torch.Tensor: The inequality residuals. Shape: (batch_size, 2 * num).

        """
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.view(1, -1)

        state_np = state.cpu().numpy()

        upper_bound = np.ones(self.num) * self.high
        lower_bound = np.ones(self.num) * self.low

        max_power = (
            1
            / self.eta_in
            * np.concatenate(
                (self.p_max *
                 np.ones((state_np.shape[0], self.num, 1)), (upper_bound -
                                                             state_np)[:, :, None]),
                axis=2,
            )
        )
        min_power = self.eta_out * np.concatenate(
            (self.p_min *
             np.ones((state_np.shape[0], self.num, 1)), (lower_bound -
                                                         state_np)[:, :, None]),
            axis=2,
        )

        # Convert max and min power back to Torch tensors
        max_power_torch = torch.tensor(
            np.min(
                max_power,
                axis=2),
            dtype=torch.get_default_dtype(),
            device=self.device)
        min_power_torch = torch.tensor(
            np.max(
                min_power,
                axis=2),
            dtype=torch.get_default_dtype(),
            device=self.device)

        # Calculate residuals
        residuals = torch.cat(
            [
                action - max_power_torch,  # Residuals for exceeding max power
                min_power_torch - action,  # Residuals for falling below min power
            ],
            dim=1,
        )

        return residuals

    def ineq_dist(self, state: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
        """
        Compute the clamped inequality residuals for the given state and action.

        This method calculates the inequality residuals using the `ineq_resid` method
        and clamps the residuals to ensure they are non-negative. The clamped residuals
        represent the distances to the inequality constraints.

        Args:
            state (torch.Tensor): The current state of the battery. Shape: (batch_size, num).
            action (torch.Tensor): The action to be applied to the battery. Shape: (batch_size, num).

        Returns:
            torch.Tensor: The clamped inequality residuals. Shape: (batch_size, 2 * num).
        """
        residuals = self.ineq_resid(state, action)

        # Clamp residuals to ensure non-negativity
        clamped_residuals = torch.clamp(residuals, min=0)

        return clamped_residuals

    def update_bound(
            self, state: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the updated power bounds for the battery based on its current state.

        This method calculates the maximum (`new_p_max`) and minimum (`new_p_min`) allowable
        power for the battery, considering its current state of charge (SOC) and operational
        constraints.

        Args:
            state (torch.Tensor): The current state of the battery. Shape: (batch_size, num + ahead).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - `new_p_max`: The updated maximum allowable power. Shape: (batch_size, num).
                - `new_p_min`: The updated minimum allowable power. Shape: (batch_size, num).
        """
        if state.dim() == 1:
            state = state.view(1, -1)

        state_np = state.cpu().numpy()

        soc_state = state_np[:, -self.num -
                             self.p_data.ahead: -self.p_data.ahead]

        upper_bound = np.ones(self.num) * self.high
        lower_bound = np.ones(self.num) * self.low

        max_power = (
            1
            / self.eta_in
            * np.concatenate(
                (self.p_max *
                 np.ones((soc_state.shape[0], self.num, 1)), (upper_bound -
                                                              soc_state)[:, :, None]),
                axis=2,
            )
        )
        min_power = self.eta_out * np.concatenate(
            (self.p_min *
             np.ones((soc_state.shape[0], self.num, 1)), (lower_bound -
                                                          soc_state)[:, :, None]),
            axis=2,
        )

        # Reduce along the last axis to get the final bounds
        new_p_max = np.min(max_power, axis=2)
        new_p_min = np.max(min_power, axis=2)

        return new_p_max, new_p_min

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Adjust the action values based on charging and discharging efficiencies.

        This method modifies the `action` array to account for the battery's charging
        (`eta_in`) and discharging (`eta_out`) efficiencies. Positive values (charging)
        are scaled by `eta_in`, and negative values (discharging) are scaled by `eta_out`.

        Args:
            action (np.ndarray): The action array to be processed. Shape: (num,).

        Returns:
            np.ndarray: The processed action array, with efficiencies applied.

        Notes:
            - The `action` array is modified in place.
            - Positive values (charging) are multiplied by `eta_in`.
            - Negative values (discharging) are divided by `eta_out`.
        """
        if not isinstance(action, np.ndarray):
            raise ValueError("The `action` parameter must be a NumPy array.")
        if not np.issubdtype(action.dtype, np.number):
            raise ValueError("The `action` array must contain numeric values.")

        action[action >= 0] *= self.eta_in  # Scale positive values (charging)
        action[action < 0] /= self.eta_out

        return action
