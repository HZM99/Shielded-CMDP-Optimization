import torch
import torch.nn as nn
from omnisafe.algorithms import FOCOPS as BaseFOCOPS
from torch.distributions import Normal

class CustomMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        
        layers = []
        sizes = [obs_dim] + list(hidden_sizes)
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        self.net = nn.Sequential(*layers)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
        nn.init.constant_(self.mu_layer.bias, 0)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        std = torch.exp(self.log_std)
        
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(torch.log(2) - pi_action - torch.nn.functional.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi

    def log_prob(self, obs, act):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        std = torch.exp(self.log_std)
        
        pi_distribution = Normal(mu, std)
        # We cannot directly compute log_prob for transformed action
        # Need to account for the transformation
        act = torch.clamp(act, -0.999999, 0.999999)
        log_prob = pi_distribution.log_prob(torch.atanh(act))
        return log_prob.sum(-1)

class CustomFOCOPS(BaseFOCOPS):
    def _init_model(self):
        super()._init_model()
        # Replace the actor with our custom implementation
        obs_dim = self._env.observation_space.shape[0]
        act_dim = self._env.action_space.shape[0]
        
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        activation = activation_map.get(
            self._cfgs.model_cfgs.actor.activation.lower(),
            nn.ReLU
        )
        self.actor = CustomMLPActor(
            obs_dim,
            act_dim,
            self._cfgs.model_cfgs.actor.hidden_sizes,
            activation
        )