import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6  # from og paper github


class ActorNetwork(nn.Module):
    """
    Actor Network
    """

    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        max_action: float,
        hidden_dims: int,
    ):
        """
        Actor/Policy Network

        Parameters
        ----------
        n_obs : int
            Dimensionality of environment observations (states).
        n_actions : int
            Dimensionality of actions.
        max_action : float
            Actions are in the continuous range [-max_action, max_action].
        hidden_dims : int
            Number of hidden dimensions to use in fully connected layers.
        """
        super().__init__()
        self.name_slug = "actor_network"

        self.n_obs = n_obs
        self.n_actions = n_actions
        self.max_action = max_action

        self.fc1 = nn.Linear(n_obs, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.mu = nn.Linear(hidden_dims, self.n_actions)
        self.ln_sigma = nn.Linear(hidden_dims, self.n_actions)

    def forward(self, states: torch.Tensor) -> torch.distributions.Distribution:
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))

        mu, ln_sigma = self.mu(x), self.ln_sigma(x)

        sigma = torch.clamp(ln_sigma, -20, 2).exp()  # og paper uses -20 and 2
        return torch.distributions.Normal(mu, sigma)

    def sample(self, states):
        # reparameterization trick: for low varaince backpropagation
        n: torch.distributions.Normal = self(states)
        u = n.rsample()

        # squashing to bound actions, then rescale: see Appendix C of [1]
        action = torch.tanh(u) * torch.tensor(self.max_action)
        log_prob = n.log_prob(u)
        log_prob -= torch.log(1 - action**2 + EPS)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic Network
    """

    def __init__(self, n_obs: int, n_actions: int, hidden_dims: int):
        super().__init__()
        self.slug_name = "critic_network"

        self.n_obs = n_obs
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(n_obs + n_actions, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.q = nn.Linear(hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class ValueNetwork(nn.Module):
    """
    Value Network
    """

    def __init__(self, n_obs: int, hidden_dims: int):
        super().__init__()
        self.slug_name = "critic_network"

        self.n_obs = n_obs
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(n_obs, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.v = nn.Linear(hidden_dims, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)
