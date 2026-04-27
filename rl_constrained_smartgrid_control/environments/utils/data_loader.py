import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pypower.idx_bus as idx_bus

from rl_constrained_smartgrid_control.environments.utils.power_flow_solver import (
    PyPowerSolver,
)


@dataclass
class DemandLoaderConfig:
    """Configuration for DemandLoader parameters."""

    rho: float = 0.5
    q_rand: bool = False
    T: int = 24
    min_pf: float = 0.9
    max_pf: float = 1.0
    regularized: bool = True


@dataclass
class DemandLoaderState:
    """State container for DemandLoader's mutable state."""

    cache: Optional[np.ndarray] = None
    counter: int = 0


class DemandLoader:
    def __init__(self, ppc, config: Optional[DemandLoaderConfig] = None):
        self.config = config if config is not None else DemandLoaderConfig()

        data_path = Path(__file__).resolve(
        ).parents[2] / "data" / "demand.pickle"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, 'rb') as f:
            self.data = np.array(pickle.load(f)["value"])

        self.total = len(self.data) // self.config.T
        self.curve = np.zeros(self.config.T)
        for i in range(self.total):
            self.curve += self.data[i *
                                    self.config.T: i *
                                    self.config.T +
                                    self.config.T]
        self.curve /= self.total

        self.baseMVA = ppc["baseMVA"]
        self.bus = ppc["bus"]
        self.gen = ppc["gen"]

        self.state = DemandLoaderState()

    def reset(self):
        """Reset the loader state and return initial values."""
        self.state.counter = 0

        if self.config.regularized:
            tmp = self.curve
        else:
            dy = np.random.randint(self.total)
            tmp = self.data[dy * self.config.T: dy *
                            self.config.T + self.config.T]

        pd, qd = self.process(tmp)

        if qd.ndim == 1:
            qd = qd[:, None]  # Reshape qd to (T, 1)

        self.state.cache = np.concatenate((pd, qd), axis=1) / self.baseMVA

        return self.fetch()

    def process(self, ps):
        """
        Process power series to generate active and reactive demand.
        """
        nbus, _ = self.bus.shape
        idx = np.nonzero(self.bus[:, idx_bus.PD])[0]
        nnz = len(idx)

        # Calculate total power and normalize
        p_total = self.bus[:, idx_bus.PD].sum() * self.config.T * 1.0
        ps = ps / ps.sum() * p_total  # (T,)
        ps = ps[:, None]  # (T, 1)

        rand_mat = np.random.dirichlet(
            np.ones(nnz), size=self.config.T)  # (T, nnz)
        p_ratio = self.bus[:, idx_bus.PD] / self.bus[:, idx_bus.PD].sum()

        p_ratio = p_ratio[None].repeat(
            self.config.T, axis=0) * (1 - self.config.rho)
        p_ratio[:, idx] += self.config.rho * rand_mat  # (T, nbus)

        pd = p_ratio * ps

        if self.config.q_rand:
            PFFactor = (
                np.random.rand(self.config.T, nbus) * (self.config.max_pf -
                                                       self.config.min_pf) + self.config.min_pf
            )
            qd = pd * np.tan(np.arccos(PFFactor)) * \
                np.sign(self.bus[:, idx_bus.QD])
        else:
            # Ensure qd matches the shape of pd
            qd = np.zeros_like(pd)
            qd[:, idx] = self.bus[idx, idx_bus.QD][None, :] * \
                np.ones((self.config.T, nnz))

        return pd, qd

    def fetch(self):
        """Fetch the next demand values and done flag."""
        done = self.state.counter >= self.config.T

        if done:
            tmp = np.zeros_like(self.state.cache[0])
        else:
            tmp = self.state.cache[self.state.counter].copy()

        self.state.counter += 1
        return tmp, done

    # Backward compatibility properties
    @property
    def T(self) -> int:
        return self.config.T

    @property
    def q_rand(self) -> bool:
        return self.config.q_rand

    @property
    def rho(self) -> float:
        return self.config.rho

    @property
    def min_pf(self) -> float:
        return self.config.min_pf

    @property
    def max_pf(self) -> float:
        return self.config.max_pf

    @property
    def regularized(self) -> bool:
        return self.config.regularized

    @property
    def counter(self) -> int:
        return self.state.counter

    @property
    def cache(self) -> Optional[np.ndarray]:
        return self.state.cache


@dataclass
class PriceLoaderConfig:
    """Configuration for PriceLoader parameters."""

    T: int = 24
    ahead: int = 24
    regularized: bool = True
    reg_bias: float = 0.1
    reg_sigma: float = 0.05


@dataclass
class PriceLoaderState:
    """State container for PriceLoader's mutable state."""

    cache: Optional[np.ndarray] = None
    counter: int = 0


class PriceLoader:
    def __init__(self, config: Optional[PriceLoaderConfig] = None):
        self.config = config if config is not None else PriceLoaderConfig()

        data_path = Path(__file__).resolve(
        ).parents[2] / "data" / "price.pickle"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, 'rb') as f:
            self.data = np.array(pickle.load(f)["SYS"])

        self.total = len(self.data) // self.config.T
        self.curve = np.zeros(self.config.T)
        for i in range(self.total):
            self.curve += self.data[i *
                                    self.config.T: i *
                                    self.config.T +
                                    self.config.T]
        self.curve /= self.total

        self.state = PriceLoaderState()

    def reset(self):
        """Reset the loader state and return initial values."""
        self.state.counter = 0

        if self.config.regularized:
            tmp = np.concatenate([self.curve, np.zeros(self.config.T)])
        else:
            dy = np.random.randint(self.total)
            tmp = self.data[dy *
                            self.config.T: dy *
                            self.config.T +
                            self.config.T +
                            self.config.ahead]

        self.state.cache = self.process(tmp)
        return self.fetch()

    def process(self, price_curve):
        """
        Process the price curve with optional regularization.
        """
        if self.config.regularized:
            mag_r = self.config.reg_bias * np.random.randn() + 1
            return mag_r * price_curve * \
                (1 + np.random.randn(len(price_curve)))
        else:
            return price_curve

    def fetch(self):
        """Fetch the next price values and done flag."""
        done = self.state.counter >= self.config.T

        if done:
            tmp = np.zeros(self.config.ahead)
        else:
            tmp = self.state.cache[self.state.counter:
                                   self.state.counter + self.config.ahead].copy()

        self.state.counter += 1
        return tmp, done

    # Backward compatibility properties
    @property
    def T(self) -> int:
        return self.config.T

    @property
    def ahead(self) -> int:
        return self.config.ahead

    @property
    def regularized(self) -> bool:
        return self.config.regularized


if __name__ == "__main__":
    solver = PyPowerSolver(network_type="ieee33")
    ppc = solver.ppc

    ex = DemandLoader(ppc=ppc)

    config = DemandLoaderConfig(rho=0.5, q_rand=True, T=24, min_pf=0.9)
    ex_custom = DemandLoader(ppc=ppc, config=config)

    test_ps = np.random.rand(24)
    pd, qd = ex.process(test_ps)
    print(f"Process result - pd shape: {pd.shape}, qd shape: {qd.shape}")

    result, done = ex.reset()
    print(f"Reset result - shape: {result.shape}, done: {done}")

    import matplotlib.pyplot as plt

    plt.plot(ex.curve)
    plt.title("Demand Curve")
    plt.show()
