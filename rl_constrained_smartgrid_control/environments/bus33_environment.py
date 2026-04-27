from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from pypower.api import makeYbus

from rl_constrained_smartgrid_control.environments.base import SmartGridEnvironment
from rl_constrained_smartgrid_control.environments.utils.battery import Battery
from rl_constrained_smartgrid_control.environments.utils.data_loader import (
    DemandLoader,
    DemandLoaderConfig,
    PriceLoader,
    PriceLoaderConfig,
)
from rl_constrained_smartgrid_control.environments.utils.power_flow_solver import (
    PyPowerSolver,
)


class IEEE33BusEnv(SmartGridEnvironment):
    """
    minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
    s.t. p_g min <= p_g <= p_g max
                              q_g min <= q_g <= q_g max
                              vmag min <= vmag <= vmag max
                              vang_slack = \theta_slack # voltage angle
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    metadata = {
        'render.modes': [
            'human',
            'rgb_array'],
        'video.frames_per_second': 50}

    def __init__(self, ineq_penalty_weight=2.0):
        self._device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.ineq_penalty_weight = ineq_penalty_weight
        self.solver = PyPowerSolver(network_type="ieee33")
        self.ppc = self.solver.ppc
        self.nahead = 24
        self.data = DemandLoader(
            ppc=self.ppc,
            config=DemandLoaderConfig(
                rho=0.5,
                q_rand=True,
                T=24,
                min_pf=0.9,
                max_pf=1.0,
                regularized=True),
        )
        self.p_data = PriceLoader(
            config=PriceLoaderConfig(
                T=24,
                ahead=self.nahead,
                regularized=True,
                reg_bias=0.1,
                reg_sigma=0.05)
        )
        self.nbus = 33
        self.ng = 1
        self.genbase = self.ppc['gen'][:, 5]
        self.baseMVA = self.ppc['baseMVA']
        self.slack = np.where(self.ppc['bus'][:, 1] == 3)[0]
        self.pv = np.where(self.ppc['bus'][:, 1] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))
        self.slack_ = np.array([np.where(x == self.spv)[0][0]
                               for x in self.slack]).astype(np.int32)
        self.pv_ = np.array([np.where(x == self.spv)[0][0]
                            for x in self.pv]).astype(np.int32)
        self.spv_ = np.array([np.where(x == self.spv)[0][0]
                             for x in self.spv]).astype(np.int32)
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        self.ne = self.ng
        self.evs = Battery(
            self.p_data,
            num=self.ne,
            genbase=self.baseMVA,
            init_strategy="full",
            device=self._device)
        self.we = 5.0
        self.wg = 1.0
        self.quad_costs = torch.tensor(
            self.ppc['gencost'][:, 4], dtype=torch.get_default_dtype(), device=self._device)
        self.lin_costs = torch.tensor(
            self.ppc['gencost'][:, 5], dtype=torch.get_default_dtype(), device=self._device)
        self.const_cost = self.ppc['gencost'][:, 6].sum()
        self.pmax = torch.tensor(
            self.ppc['gen'][:, 8] / self.genbase, dtype=torch.get_default_dtype(), device=self._device
        )
        self.pmin = torch.tensor(
            self.ppc['gen'][:, 9] / self.genbase, dtype=torch.get_default_dtype(), device=self._device
        )
        self.qmax = torch.tensor(
            self.ppc['gen'][:, 3] / self.genbase, dtype=torch.get_default_dtype(), device=self._device
        )
        self.qmin = torch.tensor(
            self.ppc['gen'][:, 4] / self.genbase, dtype=torch.get_default_dtype(), device=self._device
        )
        self.vmax = torch.tensor(
            self.ppc['bus'][:, 11], dtype=torch.get_default_dtype(), device=self._device)
        self.vmin = torch.tensor(
            self.ppc['bus'][:, 12], dtype=torch.get_default_dtype(), device=self._device)
        slackva_array = np.deg2rad(self.ppc['bus'][self.slack, 8])
        self.slack_va = torch.tensor(
            slackva_array,
            dtype=torch.get_default_dtype(),
            device=self._device)
        ppc2 = deepcopy(self.ppc)
        ppc2['bus'][:, 0] -= 1
        ppc2['branch'][:, [0, 1]] -= 1
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(
            np.real(Ybus),
            dtype=torch.get_default_dtype(),
            device=self._device)
        self.Ybusi = torch.tensor(
            np.imag(Ybus),
            dtype=torch.get_default_dtype(),
            device=self._device)
        self._xdim = 2 * self.nbus
        self._ydim = 2 * self.ng + 2 * self.nbus + self.ne
        self._neq = 2 * self.nbus
        self._nineq_grid = 4 * self.ng + 2 * self.nbus
        self._nineq = self._nineq_grid + 2 * self.ne
        self._nknowns = self.nslack
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2 * self.ng
        self.va_start_yidx = 2 * self.ng + self.nbus
        self.pe_start_yidx = 2 * self.ng + 2 * self.nbus
        action_high = np.concatenate(
            [self.pmax.cpu(), self.qmax.cpu(), self.vmax.cpu(), [np.pi]
             * self.nbus, [self.evs.p_max] * self.ne]
        )
        action_low = np.concatenate(
            [self.pmin.cpu(), self.qmin.cpu(), self.vmin.cpu(), [-np.pi]
             * self.nbus, [self.evs.p_min] * self.ne]
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32)
        self.action_low = action_low
        self.action_high = action_high
        obs_high = np.array([10.0] *
                            self.nbus +
                            [10.0] *
                            self.nbus +
                            [self.evs.high] *
                            self.ne +
                            [240.0] *
                            self.nahead)
        obs_low = np.array([-10.0] * self.nbus + [-10.0] * self.nbus +
                           [self.evs.low] * self.ne + [0.0] * self.nahead)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.eq_num = 2 * self.nbus
        self.ineq_num = 4 * self.ng + 2 * self.nbus + 2 * self.ne
        self.state = None

        self.ineq_viol_mean = 0.0
        self.ineq_viol_var = 1.0
        self.ineq_viol_count = 0
        self.grid_viol_weight = 12.0  # Increased to prioritize grid constraints
        self.ev_viol_weight = 1.0

    def step(self, action):
        pe_start = self.pe_start_yidx
        action[pe_start:] = np.clip(
            action[pe_start:], self.evs.p_min, self.evs.p_max)

        reward_grid = -self.obj_fn(action).cpu().numpy()
        reward_evs, next_state_evs = self.evs.step(action[self.pe_start_yidx:])
        reward = self.wg * reward_grid + self.we * reward_evs
        next_state_grid, done = self.data.fetch()
        eq_viol = self.eq_resid_np(self.state, action)
        ineq_resid_val = self.ineq_resid(self.state, action)
        ineq_viol_torch = torch.clamp(ineq_resid_val, min=0)
        ineq_viol = ineq_viol_torch.detach().cpu().numpy()
        info = {'ineq_viol': ineq_viol, 'eq_viol': eq_viol}

        grid_viol_sum = (
            np.sum(ineq_viol[: self._nineq_grid]) if ineq_viol.ndim > 1 else np.sum(
                ineq_viol[: self._nineq_grid])
        )
        ev_viol_sum = (
            np.sum(ineq_viol[self._nineq_grid:]) if ineq_viol.ndim > 1 else np.sum(
                ineq_viol[self._nineq_grid:])
        )
        weighted_ineq_viol = self.grid_viol_weight * \
            grid_viol_sum + self.ev_viol_weight * ev_viol_sum

        if self.ineq_viol_count == 0:
            norm_ineq = weighted_ineq_viol
        else:
            norm_ineq = (weighted_ineq_viol - self.ineq_viol_mean) / \
                (np.sqrt(self.ineq_viol_var) + 1e-8)
        reward -= self.ineq_penalty_weight * 0.4 * norm_ineq  # Reduced penalty scaling

        old_mean = self.ineq_viol_mean
        self.ineq_viol_count += 1
        self.ineq_viol_mean += (weighted_ineq_viol -
                                old_mean) / self.ineq_viol_count
        self.ineq_viol_var += (weighted_ineq_viol - old_mean) * \
            (weighted_ineq_viol - self.ineq_viol_mean)
        next_state = np.concatenate([next_state_grid, next_state_evs])
        self.state = next_state
        return next_state, reward.item(), done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state_grid, _ = self.data.reset()
        state_evs = self.evs.reset()
        self.state = np.concatenate([state_grid, state_evs])
        self.ineq_viol_mean = 0.0
        self.ineq_viol_var = 1.0
        self.ineq_viol_count = 0
        return self.state.copy(), {}

    def get_action_vars(self, action):
        pg = action[:, : self.ng]
        qg = action[:, self.ng: 2 * self.ng]
        vm = action[:, 2 * self.ng: 2 * self.ng + self.nbus]
        va = action[:, -self.ne - self.nbus: -self.ne]
        pe = action[:, -self.ne:]
        return pg, qg, vm, va, pe

    def obj_fn(self, action):
        action_tensor = (
            torch.tensor(
                action,
                device=self._device,
                dtype=torch.get_default_dtype())
            if not isinstance(action, torch.Tensor)
            else action
        )
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        pg, _, _, _, _ = self.get_action_vars(action_tensor)
        pg_mw = pg * torch.tensor(self.genbase,
                                  device=self._device,
                                  dtype=torch.get_default_dtype())
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, state, action):
        state_t = (
            torch.tensor(
                state,
                device=self._device,
                dtype=torch.get_default_dtype())
            if not torch.is_tensor(state)
            else state
        )
        action_t = (
            torch.tensor(
                action,
                device=self._device,
                dtype=torch.get_default_dtype())
            if not torch.is_tensor(action)
            else action
        )
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0)

        pg, qg, vm, va, pe = self.get_action_vars(action_t)
        vr, vi = vm * torch.cos(va), vm * torch.sin(va)
        tmp1, tmp2 = vr @ self.Ybusr - vi @ self.Ybusi, -vr @ self.Ybusi - vi @ self.Ybusr
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self._device)
        pg_expand[:, self.spv] = pg + pe
        real_resid = (pg_expand - state_t[:,
                                          : self.nbus]) - (vr * tmp1 - vi * tmp2)
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self._device)
        qg_expand[:, self.spv] = qg
        react_resid = (
            qg_expand - state_t[:, self.nbus: 2 * self.nbus]) - (vr * tmp2 + vi * tmp1)
        return torch.cat([real_resid, react_resid], dim=1) + 1e-5

    def ineq_resid(self, state, action):
        state_t = (
            torch.tensor(
                state,
                device=self._device,
                dtype=torch.get_default_dtype())
            if not torch.is_tensor(state)
            else state
        )
        action_t = (
            torch.tensor(
                action,
                device=self._device,
                dtype=torch.get_default_dtype())
            if not torch.is_tensor(action)
            else action
        )
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0)

        pg, qg, vm, va, pe = self.get_action_vars(action_t)
        resids_grid = torch.cat(
            [pg - self.pmax, self.pmin - pg, qg - self.qmax, self.qmin - qg, vm - self.vmax, self.vmin - vm], dim=1
        )
        resids_evs = self.evs.ineq_resid(
            state_t[:, 2 * self.nbus: 2 * self.nbus + self.ne], pe)
        return torch.cat([resids_grid, resids_evs], dim=1)

    def ineq_dist(self, state, action):
        return torch.clamp(self.ineq_resid(state, action), min=0)

    def ineq_dist_np(self, state, action):
        return self.ineq_dist(state, action).detach().cpu().numpy()

    def eq_resid_np(self, state, action):
        return self.eq_resid(state, action).detach().cpu().numpy()
