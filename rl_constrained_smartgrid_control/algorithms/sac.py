import random
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from rl_constrained_smartgrid_control.environments.bus33_environment import IEEE33BusEnv


class Config:
    def __init__(self):
        self.env_name = "IEEE33Bus_Improved"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42

        self.start_timesteps = 100000
        self.max_timesteps = 2_000_000  # Extended for convergence
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_freq = 4

        # SAC parameters
        self.lr = 3e-4
        self.alpha = 0.2

        self.initial_lambda_eq = 15.0  # Further increased to enforce equality constraints
        self.initial_lambda_ineq = 10.0
        self.lambda_lr_multiplier = 1.0
        self.cost_limit_eq = 0.001
        self.cost_limit_ineq = 0.001
        self.ineq_penalty_weight = 2.0

        self.log_dir = f"runs/SAC_L_{self.env_name}_{int(time.time())}"


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.eq_viol = np.zeros((max_size, 1))
        self.ineq_viol = np.zeros((max_size, 1))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.eq_viol_mean = 0.0
        self.eq_viol_var = 1.0
        self.eq_viol_count = 0
        self.ineq_viol_mean = 0.0
        self.ineq_viol_var = 1.0
        self.ineq_viol_count = 0
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

    def add(self, state, action, next_state, reward, done, info):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        raw_reward = reward
        raw_eq_viol = np.sum(np.abs(info['eq_viol']))
        raw_ineq_viol = np.sum(info['ineq_viol'])

        if self.reward_count == 0:
            self.reward_mean = raw_reward
        else:
            old_mean = self.reward_mean
            self.reward_count += 1
            self.reward_mean = old_mean + \
                (raw_reward - old_mean) / self.reward_count
            self.reward_var = self.reward_var + \
                (raw_reward - old_mean) * (raw_reward - self.reward_mean)
        norm_reward = (
            (raw_reward - self.reward_mean) /
            (np.sqrt(self.reward_var / self.reward_count) + 1e-8)
            if self.reward_count > 1
            else raw_reward
        )
        self.reward[self.ptr] = norm_reward

        if self.eq_viol_count == 0:
            self.eq_viol_mean = raw_eq_viol
        else:
            old_mean = self.eq_viol_mean
            self.eq_viol_count += 1
            self.eq_viol_mean = old_mean + \
                (raw_eq_viol - old_mean) / self.eq_viol_count
            self.eq_viol_var = self.eq_viol_var + \
                (raw_eq_viol - old_mean) * (raw_eq_viol - self.eq_viol_mean)
        norm_eq = (
            (raw_eq_viol - self.eq_viol_mean) /
            (np.sqrt(self.eq_viol_var / self.eq_viol_count) + 1e-8)
            if self.eq_viol_count > 1
            else raw_eq_viol
        )
        self.eq_viol[self.ptr] = norm_eq

        if self.ineq_viol_count == 0:
            self.ineq_viol_mean = raw_ineq_viol
        else:
            old_mean = self.ineq_viol_mean
            self.ineq_viol_count += 1
            self.ineq_viol_mean = old_mean + \
                (raw_ineq_viol - old_mean) / self.ineq_viol_count
            self.ineq_viol_var = self.ineq_viol_var + \
                (raw_ineq_viol - old_mean) * \
                (raw_ineq_viol - self.ineq_viol_mean)
        norm_ineq = (
            (raw_ineq_viol - self.ineq_viol_mean) /
            (np.sqrt(self.ineq_viol_var / self.ineq_viol_count) + 1e-8)
            if self.ineq_viol_count > 1
            else raw_ineq_viol
        )
        self.ineq_viol[self.ptr] = norm_ineq

        self.not_done[self.ptr] = 1.0 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.eq_viol[ind]).to(self.device),
            torch.FloatTensor(self.ineq_viol[ind]).to(self.device),
        )

    @property
    def reward_std(self):
        return np.sqrt(self.reward_var / self.reward_count) + 1e-8


LOG_STD_MAX, LOG_STD_MIN = 2, -20


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_l = nn.Linear(256, action_dim)
        self.log_std_l = nn.Linear(256, action_dim)
        self.max_action = torch.FloatTensor(max_action)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean_l(x)
        log_std = torch.clamp(self.log_std_l(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action.to(state.device)
        log_prob = normal.log_prob(
            x_t) - torch.log(self.max_action.to(state.device) * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1, self.l2, self.l3 = nn.Linear(
            state_dim + action_dim, 256), nn.Linear(256, 256), nn.Linear(256, 1)
        self.l4, self.l5, self.l6 = nn.Linear(
            state_dim + action_dim, 256), nn.Linear(256, 256), nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(sa))))))
        q2 = F.relu(self.l6(F.relu(self.l5(F.relu(self.l4(sa))))))
        return q1, q2


class SAC_L:
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = config.device
        self.gamma, self.tau, self.alpha, self.policy_freq = config.gamma, config.tau, config.alpha, config.policy_freq
        self.total_it = 0
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr)

        self.cost_limit_eq, self.cost_limit_ineq = config.cost_limit_eq, config.cost_limit_ineq
        self.log_lambda_eq = torch.tensor(
            np.log(config.initial_lambda_eq), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.log_lambda_ineq = torch.tensor(
            np.log(config.initial_lambda_ineq), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.lambda_optimizer = torch.optim.Adam(
            [
                {'params': self.log_lambda_eq,
                 'lr': config.lr * config.lambda_lr_multiplier},
                {'params': self.log_lambda_ineq,
                 'lr': config.lr * config.lambda_lr_multiplier},
            ]
        )

        self.lambda_min, self.lambda_max = 0.01, 100.0
        self.eq_viol_history = deque(maxlen=100)
        self.ineq_viol_history = deque(maxlen=100)

        self.prev_error_eq = 0.0
        self.prev_integral_eq = 0.0
        self.prev_error_ineq = 0.0
        self.prev_integral_ineq = 0.0
        # Increased kp for aggressive eq enforcement
        self.kp, self.ki, self.kd = 2.0, 0.01, 0.1

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size):
        self.total_it += 1
        state, action, next_state, reward, not_done, eq_viol, ineq_viol = replay_buffer.sample(
            batch_size)
        lambda_eq, lambda_ineq = torch.exp(
            self.log_lambda_eq), torch.exp(
            self.log_lambda_ineq)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - \
                self.alpha * next_log_prob
            augmented_reward = reward - lambda_eq * eq_viol - lambda_ineq * ineq_viol
            target_Q = augmented_reward + (not_done * self.gamma * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss, _ = None, None
        if self.total_it % self.policy_freq == 0:
            pi, log_pi = self.actor.sample(state)
            q1_pi, q2_pi = self.critic(state, pi)
            actor_loss = ((self.alpha * log_pi) -
                          torch.min(q1_pi, q2_pi)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.eq_viol_history.append(eq_viol.mean().item())
            self.ineq_viol_history.append(ineq_viol.mean().item())
            adaptive_limit_eq = max(
                self.cost_limit_eq, np.mean(
                    self.eq_viol_history) * 0.5)
            adaptive_limit_ineq = max(0.0005,
                                      0.5 * np.mean(self.ineq_viol_history))

            error_eq = (eq_viol - adaptive_limit_eq).mean()
            integral_eq = self.prev_integral_eq + error_eq
            derivative_eq = error_eq - self.prev_error_eq
            lambda_adjust_eq = (
                self.kp * error_eq + self.ki * integral_eq +
                self.kd * derivative_eq - 0.1 * derivative_eq
            )
            self.log_lambda_eq.data += config.lr * \
                config.lambda_lr_multiplier * lambda_adjust_eq
            self.prev_error_eq, self.prev_integral_eq = error_eq.item(), integral_eq.item()

            error_ineq = (ineq_viol - adaptive_limit_ineq).mean()
            integral_ineq = self.prev_integral_ineq + error_ineq
            derivative_ineq = error_ineq - self.prev_error_ineq
            lambda_adjust_ineq = self.kp * error_ineq + \
                self.ki * integral_ineq + self.kd * derivative_ineq
            self.log_lambda_ineq.data += (
                config.lr * config.lambda_lr_multiplier * lambda_adjust_ineq * 1.1
            )  # Adaptive multiplier
            self.prev_error_ineq, self.prev_integral_ineq = error_ineq.item(), integral_ineq.item()

            self.log_lambda_eq.data = torch.clamp(
                self.log_lambda_eq.data, np.log(
                    self.lambda_min), np.log(
                    self.lambda_max)
            )
            self.log_lambda_ineq.data = torch.clamp(
                self.log_lambda_ineq.data, np.log(
                    self.lambda_min), np.log(
                    self.lambda_max)
            )

            for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss else None,
            "lambda_eq": lambda_eq.item(),
            "lambda_ineq": lambda_ineq.item(),
        }


# --- Main Training Loop ---
if __name__ == "__main__":
    config = Config()

    env = IEEE33BusEnv(ineq_penalty_weight=config.ineq_penalty_weight)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    agent = SAC_L(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    writer = SummaryWriter(log_dir=config.log_dir)
    print(f"TensorBoard log directory: {config.log_dir}")
    print(f"Using device: {config.device}")
    state, _ = env.reset()
    episode_num = 0
    ep_rewards, ep_eq_viols, ep_ineq_viols = [], [], []
    for t in range(int(config.max_timesteps)):
        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        if t < 0.75 * config.max_timesteps:
            action = np.clip(
                action,
                env.action_low * 0.25,
                env.action_high * 0.25)

        next_state, reward, done, _, info = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done, info)
        state = next_state

        ep_rewards.append(reward)
        ep_eq_viols.append(np.sum(np.abs(info['eq_viol'])))
        ep_ineq_viols.append(np.sum(info['ineq_viol']))

        if t >= config.start_timesteps:
            losses = agent.update(replay_buffer, config.batch_size)
            if (t + 1) % 1000 == 0:
                for key, val in losses.items():
                    if val is not None:
                        writer.add_scalar(f'Loss/{key}', val, t)
        if done:
            episode_num += 1
            print(
                f"Episode: {episode_num}, Reward: {np.mean(ep_rewards):.2f}, Ineq Viol: {np.mean(ep_ineq_viols):.2f}")

            writer.add_scalar(
                'Episodic/Reward',
                np.mean(ep_rewards),
                episode_num)
            writer.add_scalar(
                'Episodic/Eq_Viol_Mean',
                np.mean(ep_eq_viols),
                episode_num)
            writer.add_scalar(
                'Episodic/Eq_Viol_Max_Inst',
                np.max(ep_eq_viols),
                episode_num)
            writer.add_scalar(
                'Episodic/Ineq_Viol_Mean',
                np.mean(ep_ineq_viols),
                episode_num)
            writer.add_scalar(
                'Episodic/Ineq_Viol_Max_Inst',
                np.max(ep_ineq_viols),
                episode_num)

            state, _ = env.reset()
            ep_rewards, ep_eq_viols, ep_ineq_viols = [], [], []
    writer.close()
