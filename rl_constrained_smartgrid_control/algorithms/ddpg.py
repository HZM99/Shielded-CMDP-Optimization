# train_ddpg_l_33bus.py
import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_constrained_smartgrid_control.environments.bus33_environment import IEEE33BusEnv


def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, device=device, dtype=torch.float32)


def fanin_init(layer):
    if isinstance(layer, nn.Linear):
        bound = 1.0 / math.sqrt(layer.weight.data.size(0))
        nn.init.uniform_(layer.weight.data, -bound, +bound)
        nn.init.uniform_(layer.bias.data, -bound, +bound)


def constraint_metrics(info: dict) -> Dict[str, float]:
    """
    Returns scalar metrics for constraints:
      - ineq_mean, ineq_max : clamp(ReLU) distances for inequality constraints
      - eq_mae, eq_max, eq_l2 : equality residual stats
    """
    ineq = info.get("ineq_viol", None)
    eq = info.get("eq_viol", None)

    if ineq is not None:
        ineq = np.asarray(ineq, dtype=np.float32).ravel()
        ineq_pos = np.maximum(ineq, 0.0)
        ineq_mean = float(np.mean(ineq_pos))
        ineq_max = float(np.max(ineq_pos)) if ineq_pos.size > 0 else 0.0
    else:
        ineq_mean = 0.0
        ineq_max = 0.0

    if eq is not None:
        eq = np.asarray(eq, dtype=np.float32).ravel()
        eq_abs = np.abs(eq)
        eq_mae = float(np.mean(eq_abs)) if eq_abs.size > 0 else 0.0
        eq_max = float(np.max(eq_abs)) if eq_abs.size > 0 else 0.0
        eq_l2 = float(np.sqrt(np.sum(eq * eq))) if eq.size > 0 else 0.0
    else:
        eq_mae = 0.0
        eq_max = 0.0
        eq_l2 = 0.0

    return dict(
        ineq_mean=ineq_mean,
        ineq_max=ineq_max,
        eq_mae=eq_mae,
        eq_max=eq_max,
        eq_l2=eq_l2,
    )


# ============================================================
#                     Networks & Buffer
# ============================================================


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), activate_out=False):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.activate_out = activate_out
        self.apply(fanin_init)

    def forward(self, x):
        x = self.net(x)
        return torch.tanh(x) if self.activate_out else x


class Actor(nn.Module):
    """
    Tanh policy mapped to per-dimension action bounds.
    """

    def __init__(self, state_dim, action_low, action_high, hidden=(256, 256)):
        super().__init__()
        self.body = MLP(
            state_dim,
            action_low.shape[0],
            hidden,
            activate_out=True)
        self.register_buffer("a_low", action_low)
        self.register_buffer("a_high", action_high)

    def forward(self, s):
        t = self.body(s)  # [-1,1]
        mid = 0.5 * (self.a_high + self.a_low)
        half = 0.5 * (self.a_high - self.a_low)
        a = mid + half * t
        return torch.max(torch.min(a, self.a_high),
                         self.a_low)  # clamp for safety


class Critic(nn.Module):
    """Single Q(s,a). Use two instances: reward critic (Q) and cost critic (Qc)."""

    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden, activate_out=False)

    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=200_000, device="cpu"):
        self.device = device
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.s = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, action_dim),
                             dtype=torch.float32, device=device)
        self.r = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.c = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.s2 = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device)
        self.d = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, s, a, r, c, s2, done):
        n = s.shape[0] if s.ndim == 2 else 1
        if n > 1:
            idxs = (torch.arange(n, device=self.device) +
                    self.ptr) % self.capacity
            self.s[idxs] = s
            self.a[idxs] = a
            self.r[idxs] = r
            self.c[idxs] = c
            self.s2[idxs] = s2
            self.d[idxs] = done
            self.ptr = (self.ptr + n) % self.capacity
            self.full = self.full or self.ptr == 0
        else:
            self.s[self.ptr] = s
            self.a[self.ptr] = a
            self.r[self.ptr] = r
            self.c[self.ptr] = c
            self.s2[self.ptr] = s2
            self.d[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.capacity
            if self.ptr == 0:
                self.full = True

    def sample(self, batch_size):
        size = self.capacity if self.full else self.ptr
        idx = torch.randint(0, size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.c[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.capacity if self.full else self.ptr


@dataclass
class DDPGLConfig:
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    cost_critic_lr: float = 1e-3
    lambda_lr: float = 5e-3  # dual ascent step
    explore_noise_std: float = 0.1
    explore_noise_clip: float = 0.5
    start_steps: int = 5_000
    updates_per_step: int = 1
    cost_limit: float = 0.0  # target expected constraint cost
    w_ineq: float = 1.0  # inequality weight in cost signal
    w_eq: float = 0.1  # equality weight in cost signal
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_grad_norm: float = 10.0


class DDPGLAgent:
    def __init__(self, env: IEEE33BusEnv, cfg: DDPGLConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        a_low = to_tensor(env.action_space.low, self.device)
        a_high = to_tensor(env.action_space.high, self.device)

        self.actor = Actor(env.state_dim, a_low, a_high).to(self.device)
        self.actor_targ = Actor(env.state_dim, a_low, a_high).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.q = Critic(
            env.state_dim,
            env.action_dim).to(
            self.device)  # reward critic
        self.q_targ = Critic(env.state_dim, env.action_dim).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())

        self.qc = Critic(
            env.state_dim,
            env.action_dim).to(
            self.device)  # cost critic
        self.qc_targ = Critic(env.state_dim, env.action_dim).to(self.device)
        self.qc_targ.load_state_dict(self.qc.state_dict())

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.qc_opt = torch.optim.Adam(
            self.qc.parameters(), lr=cfg.cost_critic_lr)

        # Lagrange multiplier λ >= 0 via softplus
        self._lambda_raw = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.lambda_opt = torch.optim.Adam(
            [self._lambda_raw], lr=cfg.lambda_lr)

        self.replay = ReplayBuffer(
            env.state_dim,
            env.action_dim,
            device=self.device)

        self.a_low = a_low
        self.a_high = a_high
        self.a_mid = 0.5 * (self.a_high + self.a_low)
        self.a_half = 0.5 * (self.a_high - self.a_low)

    @property
    def lam(self):
        return F.softplus(self._lambda_raw)

    @torch.no_grad()
    def act(self, s, noise=True):
        s = to_tensor(s, self.device).unsqueeze(
            0) if s.ndim == 1 else to_tensor(s, self.device)
        a = self.actor(s)
        if noise:
            eps = torch.clamp(
                torch.randn_like(a) * self.cfg.explore_noise_std,
                -self.cfg.explore_noise_clip,
                +self.cfg.explore_noise_clip,
            )
            a = a + eps * self.a_half  # scale noise to action span
        a = torch.max(torch.min(a, self.a_high), self.a_low)
        return a.squeeze(0).cpu().numpy()

    def compute_cost_from_info(self, info: dict) -> float:
        ineq = info.get("ineq_viol", None)
        eq = info.get("eq_viol", None)
        c_ineq = float(
            np.mean(
                np.maximum(
                    np.asarray(ineq),
                    0.0))) if ineq is not None else 0.0
        c_eq = float(np.mean(np.abs(np.asarray(eq)))
                     ) if eq is not None else 0.0
        return self.cfg.w_ineq * c_ineq + self.cfg.w_eq * c_eq

    def soft_update(self, net, targ):
        with torch.no_grad():
            for p, p_t in zip(net.parameters(), targ.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau)
                p_t.data.add_(self.cfg.tau * p.data)

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return {}

        s, a, r, c, s2, d = self.replay.sample(self.cfg.batch_size)

        with torch.no_grad():
            a2 = self.actor_targ(s2)
            q_t = self.q_targ(s2, a2)
            qc_t = self.qc_targ(s2, a2)
            y = r + (1.0 - d) * self.cfg.gamma * q_t
            yc = c + (1.0 - d) * self.cfg.gamma * qc_t

        # reward critic
        q_loss = F.mse_loss(self.q(s, a), y)
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.q_opt.step()

        # cost critic
        qc_loss = F.mse_loss(self.qc(s, a), yc)
        self.qc_opt.zero_grad()
        qc_loss.backward()
        nn.utils.clip_grad_norm_(self.qc.parameters(), self.cfg.max_grad_norm)
        self.qc_opt.step()

        # actor: minimize Q - λ * (-Qc) == minimize (-Q + λ * Qc)
        a_pi = self.actor(s)
        q_pi = self.q(s, a_pi)
        qc_pi = self.qc(s, a_pi)
        actor_loss = (-q_pi + self.lam * qc_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.cfg.max_grad_norm)
        self.actor_opt.step()

        # dual update: maximize (Jc - cost_limit) wrt λ  => grad descent on
        # -lam*(Jc-cost_limit)
        dual_obj = qc_pi.mean().detach() - self.cfg.cost_limit
        self.lambda_opt.zero_grad()
        lam = self.lam
        (-(lam * dual_obj)).backward()
        self.lambda_opt.step()

        # targets
        self.soft_update(self.q, self.q_targ)
        self.soft_update(self.qc, self.qc_targ)
        self.soft_update(self.actor, self.actor_targ)

        return {
            "q_loss": q_loss.item(),
            "qc_loss": qc_loss.item(),
            "actor_loss": actor_loss.item(),
            "lambda": lam.item(),
            "dual_obj": dual_obj.item(),
            "q_pi": q_pi.mean().item(),
            "qc_pi": qc_pi.mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, n_episodes=3):
        ret_sum = 0.0
        cost_sum = 0.0
        ineq_mean_acc = 0.0
        eq_mae_acc = 0.0
        ineq_max_over_eps = 0.0
        eq_max_over_eps = 0.0

        for _ in range(n_episodes):
            s, _ = self.env.reset()
            done = False
            e_ret = 0.0
            e_cost = 0.0
            e_ineq_sum = 0.0
            e_eq_mae_sum = 0.0
            e_ineq_max = 0.0
            e_eq_max = 0.0
            e_len = 0

            while not done:
                a = self.act(s, noise=False)
                s, r, done, _, info = self.env.step(a)
                cm = constraint_metrics(info)
                e_ret += float(r)
                e_cost += self.compute_cost_from_info(info)
                e_ineq_sum += cm["ineq_mean"]
                e_eq_mae_sum += cm["eq_mae"]
                e_ineq_max = max(e_ineq_max, cm["ineq_max"])
                e_eq_max = max(e_eq_max, cm["eq_max"])
                e_len += 1

            ret_sum += e_ret
            cost_sum += e_cost
            if e_len > 0:
                ineq_mean_acc += e_ineq_sum / e_len
                eq_mae_acc += e_eq_mae_sum / e_len
            ineq_max_over_eps = max(ineq_max_over_eps, e_ineq_max)
            eq_max_over_eps = max(eq_max_over_eps, e_eq_max)

        n = max(1, n_episodes)
        return (
            ret_sum / n,
            cost_sum / n,
            ineq_mean_acc / n,
            ineq_max_over_eps,
            eq_mae_acc / n,
            eq_max_over_eps,
        )


def save_checkpoint(path: Path, agent: DDPGLAgent,
                    step: int, best_eval: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "best_eval": best_eval,
        "actor": agent.actor.state_dict(),
        "critic": agent.q.state_dict(),
        "cost_critic": agent.qc.state_dict(),
        "actor_targ": agent.actor_targ.state_dict(),
        "critic_targ": agent.q_targ.state_dict(),
        "cost_critic_targ": agent.qc_targ.state_dict(),
        "lambda_raw": agent._lambda_raw.detach().cpu(),
        # replay (optional; comment out if too big)
        "replay_ptr": agent.replay.ptr,
        "replay_full": agent.replay.full,
        "replay_s": agent.replay.s.detach().cpu(),
        "replay_a": agent.replay.a.detach().cpu(),
        "replay_r": agent.replay.r.detach().cpu(),
        "replay_c": agent.replay.c.detach().cpu(),
        "replay_s2": agent.replay.s2.detach().cpu(),
        "replay_d": agent.replay.d.detach().cpu(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, agent: DDPGLAgent):
    ckpt = torch.load(path, map_location=agent.device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.q.load_state_dict(ckpt["critic"])
    agent.qc.load_state_dict(ckpt["cost_critic"])
    agent.actor_targ.load_state_dict(ckpt["actor_targ"])
    agent.q_targ.load_state_dict(ckpt["critic_targ"])
    agent.qc_targ.load_state_dict(ckpt["cost_critic_targ"])
    with torch.no_grad():
        agent._lambda_raw.copy_(
            torch.as_tensor(
                ckpt["lambda_raw"],
                device=agent.device))
    # restore replay
    agent.replay.ptr = ckpt["replay_ptr"]
    agent.replay.full = ckpt["replay_full"]
    agent.replay.s = ckpt["replay_s"].to(agent.device)
    agent.replay.a = ckpt["replay_a"].to(agent.device)
    agent.replay.r = ckpt["replay_r"].to(agent.device)
    agent.replay.c = ckpt["replay_c"].to(agent.device)
    agent.replay.s2 = ckpt["replay_s2"].to(agent.device)
    agent.replay.d = ckpt["replay_d"].to(agent.device)
    return ckpt.get("step", 0), ckpt.get("best_eval", -1e9)


def train(args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = IEEE33BusEnv()

    cfg = DDPGLConfig(
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        cost_critic_lr=args.cost_critic_lr,
        lambda_lr=args.lambda_lr,
        explore_noise_std=args.noise_std,
        explore_noise_clip=args.noise_clip,
        start_steps=args.start_steps,
        updates_per_step=args.updates_per_step,
        cost_limit=args.cost_limit,
        w_ineq=args.w_ineq,
        w_eq=args.w_eq,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        max_grad_norm=args.max_grad_norm,
    )

    agent = DDPGLAgent(env, cfg)

    # Logging dirs
    run_dir = Path(args.outdir) / time.strftime("%Y%m%d-%H%M%S")
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "ckpt"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Resume
    start_step = 0
    best_eval = -1e9
    if args.resume and Path(args.resume).is_file():
        start_step, best_eval = load_checkpoint(Path(args.resume), agent)
        print(
            f"Resumed from {args.resume} @ step={start_step}, best_eval={best_eval:.3f}")

    s, _ = env.reset(seed=args.seed)
    s = to_tensor(s, agent.device)

    # Episode accumulators
    ep_cnt = 0
    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0
    ep_ineq_sum = 0.0
    ep_ineq_max = 0.0
    ep_eq_mae_sum = 0.0
    ep_eq_max = 0.0
    ep_eq_l2_sum = 0.0

    for t in range(start_step + 1, args.total_steps + 1):
        # Action (exploration)
        if t < cfg.start_steps:
            a_np = np.random.uniform(
                env.action_space.low,
                env.action_space.high).astype(
                np.float32)
        else:
            a_np = agent.act(s.cpu().numpy(), noise=True)

        # Step env
        s2_np, r, done, _, info = env.step(a_np)

        # Step-level constraint metrics logging
        cm = constraint_metrics(info)
        writer.add_scalar("step/reward", r, t)
        writer.add_scalar("step/ineq/mean", cm["ineq_mean"], t)
        writer.add_scalar("step/ineq/max", cm["ineq_max"], t)
        writer.add_scalar("step/eq/mae", cm["eq_mae"], t)
        writer.add_scalar("step/eq/max", cm["eq_max"], t)
        writer.add_scalar("step/eq/l2", cm["eq_l2"], t)

        # Compose cost for DDPG-L
        c_val = agent.compute_cost_from_info(info)

        # Push to replay
        s2 = to_tensor(s2_np, agent.device)
        a = to_tensor(a_np, agent.device)
        r_t = to_tensor([r], agent.device).unsqueeze(-1)
        c_t = to_tensor([c_val], agent.device).unsqueeze(-1)
        d_t = to_tensor([float(done)], agent.device).unsqueeze(-1)
        agent.replay.push(s, a, r_t, c_t, s2, d_t)

        # Move forward
        s = s2
        ep_ret += float(r)
        ep_cost += float(c_val)
        ep_len += 1
        ep_ineq_sum += cm["ineq_mean"]
        ep_ineq_max = max(ep_ineq_max, cm["ineq_max"])
        ep_eq_mae_sum += cm["eq_mae"]
        ep_eq_max = max(ep_eq_max, cm["eq_max"])
        ep_eq_l2_sum += cm["eq_l2"]

        # Updates
        logs = {}
        if t >= cfg.start_steps:
            for _ in range(cfg.updates_per_step):
                logs = agent.update()
                if logs:
                    writer.add_scalar("loss/q", logs["q_loss"], t)
                    writer.add_scalar("loss/qc", logs["qc_loss"], t)
                    writer.add_scalar("loss/actor", logs["actor_loss"], t)
                    writer.add_scalar("lagrange/lambda", logs["lambda"], t)
                    writer.add_scalar(
                        "diagnostics/dual_obj", logs["dual_obj"], t)
                    writer.add_scalar("diagnostics/q_pi", logs["q_pi"], t)
                    writer.add_scalar("diagnostics/qc_pi", logs["qc_pi"], t)

        # Episode end
        if done:
            ep_cnt += 1
            if ep_len > 0:
                writer.add_scalar("episode/return", ep_ret, t)
                writer.add_scalar("episode/cost", ep_cost, t)
                writer.add_scalar("episode/length", ep_len, t)
                writer.add_scalar("episode/ineq/mean", ep_ineq_sum / ep_len, t)
                writer.add_scalar("episode/ineq/max", ep_ineq_max, t)
                writer.add_scalar("episode/eq/mae", ep_eq_mae_sum / ep_len, t)
                writer.add_scalar("episode/eq/max", ep_eq_max, t)
                writer.add_scalar(
                    "episode/eq/l2_mean", ep_eq_l2_sum / ep_len, t)

            print(
                f"[{t:>7}] ep {ep_cnt:>4}  R: {ep_ret:>10.3f}  C: {ep_cost:>10.6f}  "
                f"L: {ep_len:>4}  λ: {logs.get('lambda', float('nan')):.5f}"
            )

            # reset episode accumulators
            s, _ = env.reset()
            s = to_tensor(s, agent.device)
            ep_ret = ep_cost = 0.0
            ep_len = 0
            ep_ineq_sum = ep_ineq_max = 0.0
            ep_eq_mae_sum = ep_eq_max = 0.0
            ep_eq_l2_sum = 0.0

        # Evaluation
        if t % args.eval_every == 0:
            eval_ret, eval_cost, ineq_mean, ineq_max, eq_mae, eq_max = agent.evaluate(
                n_episodes=args.eval_episodes)
            writer.add_scalar("eval/avg_return", eval_ret, t)
            writer.add_scalar("eval/avg_cost", eval_cost, t)
            writer.add_scalar("eval/ineq/mean", ineq_mean, t)
            writer.add_scalar("eval/ineq/max", ineq_max, t)
            writer.add_scalar("eval/eq/mae", eq_mae, t)
            writer.add_scalar("eval/eq/max", eq_max, t)
            writer.add_scalar("lagrange/lambda_eval", agent.lam.item(), t)

            print(
                f"[EVAL {t}] avgR={eval_ret:.3f} avgC={eval_cost:.6e} "
                f"ineq(mean/max)={ineq_mean:.3e}/{ineq_max:.3e} "
                f"eq(mae/max)={eq_mae:.3e}/{eq_max:.3e} "
                f"λ={agent.lam.item():.5f}"
            )

            # Save best
            if eval_ret > best_eval:
                best_eval = eval_ret
                save_checkpoint(ckpt_dir / "best.pt", agent, t, best_eval)

        # Periodic checkpoint
        if t % args.ckpt_every == 0:
            save_checkpoint(ckpt_dir / f"step_{t}.pt", agent, t, best_eval)

    # Final save
    save_checkpoint(ckpt_dir / "final.pt", agent, args.total_steps, best_eval)
    writer.close()


def make_parser():
    p = argparse.ArgumentParser(
        "DDPG-L for IEEE33BusEnv with full TensorBoard logging & checkpoints")
    p.add_argument("--total_steps", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--cost_critic_lr", type=float, default=1e-3)
    p.add_argument("--lambda_lr", type=float, default=5e-3)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--start_steps", type=int, default=5_000)
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--cost_limit", type=float, default=0.0)
    p.add_argument("--w_ineq", type=float, default=1.0)
    p.add_argument("--w_eq", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--eval_every", type=int, default=5_000)
    p.add_argument("--eval_episodes", type=int, default=3)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--outdir", type=str, default="runs_ddpg_l")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    train(args)
