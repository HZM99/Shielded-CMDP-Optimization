# rl_constrained_smartgrid_control/algorithms/cpo.py
# Batch CPO with constraint scaling + robust line search
# Logs remain unscaled; optimization uses scaled costs.

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from rl_constrained_smartgrid_control.environments.bus33_environment import IEEE33BusEnv


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), activation=nn.Tanh):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianTanhPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(256, 256)):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.register_buffer(
            "a_low", torch.as_tensor(
                act_low, dtype=torch.float32))
        self.register_buffer(
            "a_high", torch.as_tensor(
                act_high, dtype=torch.float32))
        self.register_buffer("degenerate", (self.a_high - self.a_low) <= 1e-8)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def _squash(self, z):
        t = torch.tanh(z)
        scaled = self.a_low + 0.5 * (t + 1.0) * (self.a_high - self.a_low)
        if self.degenerate.any():
            scaled = torch.where(self.degenerate, self.a_low, scaled)
        return scaled

    def act(self, obs, deterministic=False):
        mu, std = self(obs)
        z = mu if deterministic else mu + std * torch.randn_like(mu)
        a = self._squash(z)
        return a, z, mu, std

    @staticmethod
    def log_prob_pre_tanh(z, mu, std):
        return Normal(mu, std).log_prob(z).sum(-1)

    @staticmethod
    def kl_pre_tanh(mu0, std0, mu1, std1):
        var0, var1 = std0**2, std1**2
        t1 = torch.log(std1 / std0)
        t2 = (var0 + (mu0 - mu1) ** 2) / (2.0 * var1)
        t3 = -0.5
        return (t1 + t2 + t3).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden)

    def forward(self, x):
        return self.v(x).squeeze(-1)


def gae(signal: np.ndarray, values: np.ndarray, masks: np.ndarray,
        gamma: float, lam: float) -> np.ndarray:
    adv = np.zeros_like(signal)
    last = 0.0
    for t in reversed(range(len(signal))):
        next_v = values[t + 1] if t + 1 < len(signal) else 0.0
        delta = signal[t] + gamma * next_v * masks[t] - values[t]
        last = delta + gamma * lam * masks[t] * last
        adv[t] = last
    return adv


def conjugate_gradients(Avp, b, nsteps=20, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rsold = torch.dot(r, r)
    for _ in range(nsteps):
        Ap = Avp(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        if rsnew < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in model.parameters()])


def set_params_from_flat(model: nn.Module, flat: torch.Tensor):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i: i + n].view_as(p))
        i += n


def flat_grad(
    y: torch.Tensor, params: List[torch.nn.Parameter], retain_graph: bool = False, create_graph: bool = False
) -> torch.Tensor:
    grads = torch.autograd.grad(
        y,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph)
    return torch.cat([gi.contiguous().view(-1) for gi in grads])


def collect_batch(env: gym.Env, policy: GaussianTanhPolicy,
                  steps_per_iter: int, gamma: float, device: torch.device):
    obs_buf, act_buf, z_buf, mu_buf, std_buf = [], [], [], [], []
    rew_buf, ceq_buf, cineq_buf, done_buf = [], [], [], []
    ep_returns, ep_lens = [], []
    ep_c_eq_disc, ep_c_ineq_disc = [], []
    ep_max_eq_inst, ep_max_ineq_inst = [], []

    steps = 0
    o, _ = env.reset()
    disc_eq = 0.0
    disc_in = 0.0
    disc = 1.0
    max_eq = 0.0
    max_in = 0.0
    ep_ret = 0.0
    ep_len = 0

    while steps < steps_per_iter:
        ot = torch.as_tensor(
            o,
            dtype=torch.float32,
            device=device).unsqueeze(0)
        with torch.no_grad():
            a, z, mu, std = policy.act(ot)
        a_np = a.squeeze(0).cpu().numpy()
        z_np = z.squeeze(0).cpu().numpy()
        mu_np = mu.squeeze(0).cpu().numpy()
        std_np = std.squeeze(0).cpu().numpy()

        o2, r, d, trunc, info = env.step(a_np)
        done_flag = bool(d or trunc)

        eq_viol = info.get("eq_viol", None)
        ineq_viol = info.get("ineq_viol", None)

        # Unscaled costs for logging
        c_eq = float(np.linalg.norm(np.array(eq_viol).reshape(-1),
                     ord=2)) if eq_viol is not None else 0.0
        if ineq_viol is None:
            c_ineq = 0.0
        else:
            ineq = np.array(ineq_viol).reshape(-1)
            c_ineq = float(np.maximum(ineq, 0.0).sum())

        obs_buf.append(o.copy())
        act_buf.append(a_np.copy())
        z_buf.append(z_np.copy())
        mu_buf.append(mu_np.copy())
        std_buf.append(std_np.copy())
        rew_buf.append(float(r))
        ceq_buf.append(c_eq)
        cineq_buf.append(c_ineq)
        done_buf.append(float(done_flag))

        ep_ret += r
        ep_len += 1
        disc_eq += disc * c_eq
        disc_in += disc * c_ineq
        max_eq = max(max_eq, c_eq)
        max_in = max(max_in, c_ineq)
        disc *= gamma

        o = o2
        steps += 1

        if done_flag:
            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)
            ep_c_eq_disc.append(disc_eq)
            ep_c_ineq_disc.append(disc_in)
            ep_max_eq_inst.append(max_eq)
            ep_max_ineq_inst.append(max_in)

            # reset episode stats
            o, _ = env.reset()
            disc_eq = 0.0
            disc_in = 0.0
            disc = 1.0
            max_eq = 0.0
            max_in = 0.0
            ep_ret = 0.0
            ep_len = 0

    batch = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "acts": np.asarray(act_buf, dtype=np.float32),
        "z_pre": np.asarray(z_buf, dtype=np.float32),
        "mu": np.asarray(mu_buf, dtype=np.float32),
        "std": np.asarray(std_buf, dtype=np.float32),
        "rews": np.asarray(rew_buf, dtype=np.float32),
        "c_eq": np.asarray(ceq_buf, dtype=np.float32),  # UNscaled
        "c_ineq": np.asarray(cineq_buf, dtype=np.float32),  # UNscaled
        "dones": np.asarray(done_buf, dtype=np.float32),
        "ep_metrics": {
            "ret": ep_returns,
            "len": ep_lens,
            "disc_eq": ep_c_eq_disc,  # UNscaled
            "disc_ineq": ep_c_ineq_disc,  # UNscaled
            "max_eq": ep_max_eq_inst,  # UNscaled
            "max_ineq": ep_max_ineq_inst,  # UNscaled
        },
    }
    return batch


@dataclass
class CPOConfig:
    seed: int = 0
    total_iters: int = 300
    steps_per_iter: int = 4096
    gamma: float = 0.995
    lam_adv: float = 0.95
    lam_cost: float = 0.95
    target_kl: float = 0.03
    cg_iters: int = 20
    backtrack_iters: int = 20
    backtrack_coef: float = 0.8
    damping: float = 1e-2
    d_eq: float = 0.0
    d_ineq: float = 0.0
    v_lr: float = 3e-4
    v_updates: int = 80
    v_batch: int = 256
    normalize_cost_advs: bool = True
    eq_scale: float = 1.0 / 1000.0
    ineq_scale: float = 1.0
    feas_tol_abs: float = 1e-4  # absolute epsilon (scaled units)
    # 0.5% improvement when infeasible (scaled units)
    infeas_improve_frac: float = 0.005
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"cpo-ieee33-{int(time.time())}"


class CPOAgent:
    def __init__(self, env: gym.Env, cfg: CPOConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_low = env.action_space.low.astype(np.float32)
        act_high = env.action_space.high.astype(np.float32)

        self.policy = GaussianTanhPolicy(
            obs_dim, act_dim, act_low, act_high).to(
            self.device)
        self.v_r = ValueNet(obs_dim).to(self.device)
        self.v_ceq = ValueNet(obs_dim).to(self.device)
        self.v_cineq = ValueNet(obs_dim).to(self.device)

        self.v_r_opt = optim.Adam(self.v_r.parameters(), lr=cfg.v_lr)
        self.v_ceq_opt = optim.Adam(self.v_ceq.parameters(), lr=cfg.v_lr)
        self.v_cineq_opt = optim.Adam(self.v_cineq.parameters(), lr=cfg.v_lr)

        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.run_name))
        self.last_accepted_kl = 0.0

    def compute_advantages(
            self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs = batch["obs"]
        rews = batch["rews"]

        ceq_scaled = batch["c_eq"] * self.cfg.eq_scale
        cineq_scaled = batch["c_ineq"] * self.cfg.ineq_scale
        dones = batch["dones"]
        masks = 1.0 - dones

        with torch.no_grad():
            v_r = self.v_r(
                torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=self.device)).cpu().numpy()
            v_ceq = self.v_ceq(
                torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=self.device)).cpu().numpy()
            v_cineq = self.v_cineq(
                torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=self.device)).cpu().numpy()

        adv_r = gae(rews, v_r, masks, self.cfg.gamma, self.cfg.lam_adv)

        adv_ceq = gae(
            ceq_scaled,
            v_ceq,
            masks,
            self.cfg.gamma,
            self.cfg.lam_cost)
        adv_cineq = gae(
            cineq_scaled,
            v_cineq,
            masks,
            self.cfg.gamma,
            self.cfg.lam_cost)

        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        if self.cfg.normalize_cost_advs:
            adv_ceq = adv_ceq / (adv_ceq.std() + 1e-8)
            adv_cineq = adv_cineq / (adv_cineq.std() + 1e-8)

        ret_r = adv_r + v_r
        ret_ceq = adv_ceq + v_ceq
        ret_cineq = adv_cineq + v_cineq

        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            "acts": torch.as_tensor(batch["acts"], dtype=torch.float32, device=self.device),
            "z_pre": torch.as_tensor(batch["z_pre"], dtype=torch.float32, device=self.device),
            "mu_old": torch.as_tensor(batch["mu"], dtype=torch.float32, device=self.device),
            "std_old": torch.as_tensor(batch["std"], dtype=torch.float32, device=self.device),
            "adv_r": torch.as_tensor(adv_r, dtype=torch.float32, device=self.device),
            "adv_ceq": torch.as_tensor(adv_ceq, dtype=torch.float32, device=self.device),
            "adv_cineq": torch.as_tensor(adv_cineq, dtype=torch.float32, device=self.device),
            "ret_r": torch.as_tensor(ret_r, dtype=torch.float32, device=self.device),
            "ret_ceq": torch.as_tensor(ret_ceq, dtype=torch.float32, device=self.device),
            "ret_cineq": torch.as_tensor(ret_cineq, dtype=torch.float32, device=self.device),
        }

    def build_fvp(self, obs, mu_old, std_old):
        params = list(self.policy.parameters())

        def fvp_func(v_flat: torch.Tensor) -> torch.Tensor:
            v_flat = v_flat.detach()
            mu, std = self.policy(obs)
            kl = self.policy.kl_pre_tanh(mu_old, std_old, mu, std).mean()
            grads = torch.autograd.grad(kl, params, create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])
            grad_v = (flat_grads * v_flat).sum()
            hv = torch.autograd.grad(grad_v, params, retain_graph=False)
            hv_flat = torch.cat([h.contiguous().view(-1) for h in hv]).detach()
            return hv_flat + self.cfg.damping * v_flat

        return fvp_func

    def single_grad(
            self, data: Dict[str, torch.Tensor], which: str) -> torch.Tensor:
        params = list(self.policy.parameters())
        obs = data["obs"]
        z_pre = data["z_pre"]
        mu_old = data["mu_old"].detach()
        std_old = data["std_old"].detach()

        mu, std = self.policy(obs)
        logp = self.policy.log_prob_pre_tanh(z_pre, mu, std)
        with torch.no_grad():
            logp_old = self.policy.log_prob_pre_tanh(z_pre, mu_old, std_old)
        ratio = torch.exp(logp - logp_old)

        if which == "obj":
            adv = data["adv_r"]
        elif which == "c_eq":
            adv = data["adv_ceq"]
        elif which == "c_ineq":
            adv = data["adv_cineq"]
        else:
            raise ValueError

        scalar = (ratio * adv).mean()
        g = flat_grad(
            scalar,
            params,
            retain_graph=False,
            create_graph=False).detach()
        return g

    def cpo_step(self, data: Dict[str, torch.Tensor],
                 costs_at_pi_mean_scaled: np.ndarray):
        th_old = flat_params(self.policy).detach()
        fvp = self.build_fvp(data["obs"], data["mu_old"], data["std_old"])

        g = self.single_grad(data, "obj")
        b1 = self.single_grad(data, "c_eq")
        b2 = self.single_grad(data, "c_ineq")

        B = torch.stack([b1, b2], dim=1)  # [n_params, 2]
        c_vec = torch.as_tensor(
            costs_at_pi_mean_scaled,
            dtype=torch.float32,
            device=g.device)

        def Hinv(x):
            return conjugate_gradients(fvp, x, nsteps=self.cfg.cg_iters)

        Hinv_g = Hinv(g)
        Hinv_B = torch.stack([Hinv(B[:, i]) for i in range(B.shape[1])], dim=1)
        r = g @ Hinv_B
        S = B.t() @ Hinv_B
        q = (g @ Hinv_g).item()
        delta = self.cfg.target_kl

        def dual_and_nu(lmbd: float):
            S_np = S.detach().cpu().numpy()
            r_np = r.detach().cpu().numpy()
            c_np = c_vec.detach().cpu().numpy()
            lam = max(lmbd, 1e-12)
            P = S_np / lam
            qv = -(r_np / lam + c_np)
            best_val = -1e30
            best_nu = np.zeros_like(qv)
            candidates = [
                np.array([1, 1], dtype=bool),
                np.array([1, 0], dtype=bool),
                np.array([0, 1], dtype=bool),
                np.array([0, 0], dtype=bool),
            ]
            for active in candidates:
                if active.sum() == 0:
                    nu = np.zeros_like(qv)
                else:
                    Paa = P[np.ix_(active, active)]
                    qaa = qv[active]
                    try:
                        nu_a = - \
                            np.linalg.solve(
                                Paa + 1e-10 * np.eye(Paa.shape[0]), qaa)
                    except np.linalg.LinAlgError:
                        continue
                    nu = np.zeros_like(qv)
                    nu[active] = nu_a
                nu = np.maximum(nu, 0.0)
                rt_nu = r_np @ nu
                nuSnu = nu @ (S_np @ nu)
                dual_val = -(q - 2 * rt_nu + nuSnu) / (2.0 * lam) + \
                    (c_np @ nu) - (lam * delta) / 2.0
                if dual_val > best_val:
                    best_val = dual_val
                    best_nu = nu
            return best_val, torch.as_tensor(
                best_nu, dtype=torch.float32, device=g.device)

        lam_candidates = np.logspace(-6, 6, num=25)
        best_val = -1e30
        lam_star, nu_star = None, None
        for lmbd in lam_candidates:
            val, nu = dual_and_nu(lmbd)
            if val > best_val:
                best_val = val
                lam_star, nu_star = lmbd, nu

        if lam_star is None:
            worst = int(np.argmax(c_vec.detach().cpu().numpy()))
            Hinv_b = Hinv(B[:, worst].detach())
            denom = (B[:, worst] * Hinv_b).sum().item() + 1e-12
            step = -math.sqrt(2.0 * delta / denom) * Hinv_b
        else:
            g_eff = g - (B @ nu_star)
            Hinv_g_eff = Hinv(g_eff)
            step = (1.0 / lam_star) * Hinv_g_eff

        def surr_and_cons_kl(th_flat: torch.Tensor):
            set_params_from_flat(self.policy, th_flat)
            mu, std = self.policy(data["obs"])
            logp = self.policy.log_prob_pre_tanh(data["z_pre"], mu, std)
            with torch.no_grad():
                logp_old = self.policy.log_prob_pre_tanh(
                    data["z_pre"], data["mu_old"], data["std_old"])
            ratio = torch.exp(logp - logp_old)

            surr_new = (ratio * data["adv_r"]).mean().item()
            c1_new = (ratio * data["adv_ceq"]).mean().item()
            c2_new = (ratio * data["adv_cineq"]).mean().item()
            kl = self.policy.kl_pre_tanh(
                data["mu_old"], data["std_old"], mu, std).mean().item()
            cons_terms_scaled = [
                costs_at_pi_mean_scaled[0] + c1_new,
                costs_at_pi_mean_scaled[1] + c2_new]
            return surr_new, cons_terms_scaled, kl

        th0 = th_old.clone()
        c0 = torch.as_tensor(
            costs_at_pi_mean_scaled,
            dtype=torch.float32,
            device=g.device)
        start_infeasible = (c0 > self.cfg.feas_tol_abs).any().item()
        agg0 = float(torch.clamp(c0, min=0.0).sum().item())
        accepted_kl = 0.0
        success = False

        for j in range(self.cfg.backtrack_iters):
            coef = self.cfg.backtrack_coef**j
            th_try = th0 + coef * step
            surr_val, cons_terms_scaled, kl_val = surr_and_cons_kl(th_try)

            ok_kl = kl_val <= self.cfg.target_kl * 1.5
            if start_infeasible:
                agg_new = sum(max(ct, 0.0) for ct in cons_terms_scaled)
                rel_gate = agg_new <= (
                    1.0 - self.cfg.infeas_improve_frac) * agg0 + 1e-12
                abs_gate = agg_new <= agg0 - self.cfg.feas_tol_abs
                accept = ok_kl and (rel_gate or abs_gate)
            else:
                accept = ok_kl and all(
                    ct <= self.cfg.feas_tol_abs for ct in cons_terms_scaled)

            if accept:
                set_params_from_flat(self.policy, th_try)
                success = True
                accepted_kl = kl_val
                break

        if not success:
            set_params_from_flat(self.policy, th0)
            accepted_kl = 0.0

        self.last_accepted_kl = accepted_kl

    def fit_values(self, data: Dict[str, torch.Tensor]):
        obs = data["obs"]
        datasets = [
            (self.v_r, self.v_r_opt, data["ret_r"]),
            (self.v_ceq, self.v_ceq_opt, data["ret_ceq"]),
            (self.v_cineq, self.v_cineq_opt, data["ret_cineq"]),
        ]
        for net, opt, target in datasets:
            for _ in range(self.cfg.v_updates):
                n = obs.shape[0]
                bs = min(self.cfg.v_batch, n)
                idx = torch.randint(0, n, (bs,), device=obs.device)
                pred = net(obs[idx])
                loss = ((pred - target[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def train(self):
        for it in range(self.cfg.total_iters):
            batch = collect_batch(
                self.env,
                self.policy,
                self.cfg.steps_per_iter,
                self.cfg.gamma,
                self.device)
            data = self.compute_advantages(batch)

            epm = batch["ep_metrics"]
            J_ceq_unscaled = float(
                np.mean(epm["disc_eq"])) if epm["disc_eq"] else 0.0
            J_cineq_unscaled = float(
                np.mean(epm["disc_ineq"])) if epm["disc_ineq"] else 0.0

            costs_at_pi_mean_scaled = np.array(
                [
                    J_ceq_unscaled * self.cfg.eq_scale - self.cfg.d_eq * self.cfg.eq_scale,
                    J_cineq_unscaled * self.cfg.ineq_scale - self.cfg.d_ineq * self.cfg.ineq_scale,
                ],
                dtype=np.float32,
            )

            self.cpo_step(data, costs_at_pi_mean_scaled)
            self.fit_values(data)

            if len(epm["ret"]) > 0:
                self.writer.add_scalar(
                    "episode/return",
                    np.mean(
                        epm["ret"]),
                    it)
                self.writer.add_scalar("episode/len", np.mean(epm["len"]), it)
                self.writer.add_scalar(
                    "constraints/ep_discounted_eq", J_ceq_unscaled, it)
                self.writer.add_scalar(
                    "constraints/ep_discounted_ineq", J_cineq_unscaled, it)
                self.writer.add_scalar(
                    "constraints/max_instant_eq",
                    np.mean(
                        epm["max_eq"]),
                    it)
                self.writer.add_scalar(
                    "constraints/max_instant_ineq",
                    np.mean(
                        epm["max_ineq"]),
                    it)

            self.writer.add_scalar("diagnostics/kl", self.last_accepted_kl, it)

            print(
                f"[{it}] R={np.mean(epm['ret']) if epm['ret'] else 0:.2f} | "
                f"Ceq={J_ceq_unscaled:.3f} | Cineq={J_cineq_unscaled:.3f} | "
                f"max(eq)={np.mean(epm['max_eq']) if epm['max_eq'] else 0:.3f} "
                f"max(ineq)={np.mean(epm['max_ineq']) if epm['max_ineq'] else 0:.3f} | "
                f"KL={self.last_accepted_kl:.4f}"
            )


def main():
    cfg = CPOConfig()
    set_seed(cfg.seed)
    env = IEEE33BusEnv(ineq_penalty_weight=2.0)
    agent = CPOAgent(env, cfg)
    agent.train()


if __name__ == "__main__":
    main()
