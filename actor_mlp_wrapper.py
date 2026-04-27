import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorMLPWrapper(nn.Module):
    def __init__(self, hidden_sizes, observation_space, action_space, activation):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = torch.as_tensor(
            (action_space.high - action_space.low) / 2.0,
            dtype=torch.float32
        )
        act_bias = torch.as_tensor(
            (action_space.high + action_space.low) / 2.0,
            dtype=torch.float32
        )
        
        # Set up network architecture
        layers = []
        in_size = obs_dim
        for next_size in hidden_sizes:
            layers.append(nn.Linear(in_size, next_size))
            layers.append(activation())
            in_size = next_size
        
        # Mean and log_std heads
        self.net = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.act_bias = act_bias
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
        nn.init.constant_(self.mu_layer.bias, 0)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        else:
            logp_pi = None
            
        # Scale and shift samples to correct range
        pi_action = self.act_limit * torch.tanh(pi_action) + self.act_bias
        
        return pi_action, logp_pi
    
    def get_logprob(self, obs, actions):
        # Invert the scaling and shifting
        unscaled_actions = torch.atanh((actions - self.act_bias) / self.act_limit)
        
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        pi_distribution = Normal(mu, std)
        return pi_distribution.log_prob(unscaled_actions).sum(axis=-1)
    
    def step(self, obs):
        with torch.no_grad():
            a, logp = self.forward(obs)
            return a, logp