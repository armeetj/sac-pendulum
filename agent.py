from pathlib import Path
from buffer import ReplayBuffer, Frame
from models import ActorNetwork, CriticNetwork, ValueNetwork
import torch.optim.adam as optim
import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        max_action: float,
        checkpoint_dir: Path | str,
        buffer_size: int = 1_000_000,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dims: int = 256,
        reward_scale: int = 2,
        batch_size: int = 256,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.max_action = max_action
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.hidden_dims = hidden_dims
        self.reward_scale = reward_scale

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = ActorNetwork(n_obs, n_actions, max_action, hidden_dims).to(
            self.device
        )
        self.critic_1 = CriticNetwork(n_obs, n_actions, hidden_dims).to(self.device)
        self.critic_2 = CriticNetwork(n_obs, n_actions, hidden_dims).to(self.device)
        self.value = ValueNetwork(n_obs, hidden_dims).to(self.device)
        self.value_target = ValueNetwork(n_obs, hidden_dims).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.update_network_parameters(1)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        actions, _ = self.actor.sample(state)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(Frame(state, action, reward, next_state, done))

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(
            self.value_target.parameters(), self.value.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = zip(
            *self.buffer.sample(self.batch_size)
        )

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device).unsqueeze(1)

        value = self.value(states).view(-1)
        next_value = self.value_target(next_states).view(-1)
        next_value = next_value * (1 - dones).view(-1)
        actions_new, log_probs = self.actor.sample(states)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, actions_new).view(-1)
        q2_new_policy = self.critic_2(states, actions_new).view(-1)
        critic_value = torch.min(q1_new_policy, q2_new_policy)

        self.value_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = F.mse_loss(value, value_target)
        value_loss.backward()
        self.value_optimizer.step()

        actions_new, log_probs = self.actor.sample(states)
        q1_new_policy = self.critic_1(states, actions_new).view(-1)
        q2_new_policy = self.critic_2(states, actions_new).view(-1)
        critic_value = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = (self.alpha * log_probs - critic_value).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        q_hat = self.reward_scale * rewards + self.gamma * next_value.unsqueeze(1)
        q1_old_policy = self.critic_1(states, actions).view(-1)
        q2_old_policy = self.critic_2(states, actions).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, q_hat.view(-1))
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat.view(-1))

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.update_network_parameters()

    def save_models(self):
        print("saving models\n===================================")
        torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/actor.pth")
        torch.save(self.value.state_dict(), f"{self.checkpoint_dir}/value.pth")
        torch.save(
            self.value_target.state_dict(), f"{self.checkpoint_dir}/value_target.pth"
        )
        torch.save(self.critic_1.state_dict(), f"{self.checkpoint_dir}/critic_1.pth")
        torch.save(self.critic_2.state_dict(), f"{self.checkpoint_dir}/critic_2.pth")

        torch.save(
            self.actor_optimizer.state_dict(),
            f"{self.checkpoint_dir}/actor_optimizer.pth",
        )
        torch.save(
            self.value_optimizer.state_dict(),
            f"{self.checkpoint_dir}/value_optimizer.pth",
        )
        torch.save(
            self.critic_1_optimizer.state_dict(),
            f"{self.checkpoint_dir}/critic_1_optimizer.pth",
        )
        torch.save(
            self.critic_2_optimizer.state_dict(),
            f"{self.checkpoint_dir}/critic_2_optimizer.pth",
        )

    def load_models(self):
        print("loading models\n===================================")
        self.actor.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/actor.pth", weights_only=True)
        )
        self.value.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/value.pth", weights_only=True)
        )
        self.value_target.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/value_target.pth", weights_only=True),
        )
        self.critic_1.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/critic_1.pth", weights_only=True)
        )
        self.critic_2.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/critic_2.pth", weights_only=True)
        )

        self.actor_optimizer.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/actor_optimizer.pth", weights_only=True)
        )
        self.value_optimizer.load_state_dict(
            torch.load(f"{self.checkpoint_dir}/value_optimizer.pth", weights_only=True)
        )
        self.critic_1_optimizer.load_state_dict(
            torch.load(
                f"{self.checkpoint_dir}/critic_1_optimizer.pth", weights_only=True
            )
        )
        self.critic_2_optimizer.load_state_dict(
            torch.load(
                f"{self.checkpoint_dir}/critic_2_optimizer.pth", weights_only=True
            )
        )
