import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MAPPOActor(nn.Module):
    """
    Shared actor network used by all agents.

    Takes a single agent's local observation (8 values) and outputs
    action logits. Weights are shared across all agents -- one policy
    governs the entire fleet.

    Architecture: obs(8) -> Linear(128) -> Tanh -> Linear(128) -> Tanh
                  -> Linear(2) [logits]

    Args:
        obs_dim: Observation dimension per agent. Default 8.
        act_dim: Number of discrete actions. Default 2.
        hidden:  Hidden layer size. Default 128.
    """

    def __init__(self, obs_dim=8, act_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_dist(self, obs):
        return torch.distributions.Categorical(logits=self.forward(obs))


class MAPPOCritic(nn.Module):
    """
    Centralised critic network.

    Takes the global state (all agents' observations concatenated) and
    outputs a scalar value estimate. Used only during training (CTDE:
    Centralised Training, Decentralised Execution).

    Architecture: global(72) -> Linear(256) -> Tanh -> Linear(256) -> Tanh
                  -> Linear(1) [value]

    Args:
        global_dim: Global state dimension. Default 72 (9 agents x 8 obs).
        hidden:     Hidden layer size. Default 256.
    """

    def __init__(self, global_dim=72, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


class MAPPOTrainer:
    """
    MAPPO trainer with Generalised Advantage Estimation (GAE).

    Implements Centralised Training with Decentralised Execution (CTDE):
        - Actor uses only local observations (deployable without global state).
        - Critic uses global state during training to reduce variance and
          resolve the non-stationarity problem of multi-agent environments.

    Args:
        n_agents:    Number of agents. Default 9.
        obs_dim:     Per-agent observation dimension. Default 8.
        act_dim:     Number of discrete actions per agent. Default 2.
        global_dim:  Global state dimension. Default 72.
        lr:          Learning rate for both actor and critic. Default 3e-4.
        gamma:       Discount factor. Default 0.99.
        gae_lambda:  GAE smoothing parameter. Default 0.95.
        clip_eps:    PPO clipping epsilon. Default 0.2.
        ent_coef:    Entropy bonus coefficient. Default 0.01.
        n_epochs:    Gradient update epochs per rollout. Default 10.
        device:      Torch device string. Default "cpu".
    """

    def __init__(self, n_agents=9, obs_dim=8, act_dim=2, global_dim=72,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, ent_coef=0.01, n_epochs=10, device="cpu"):
        self.n_agents  = n_agents
        self.gamma     = gamma
        self.lam       = gae_lambda
        self.clip_eps  = clip_eps
        self.ent_coef  = ent_coef
        self.n_epochs  = n_epochs
        self.device    = device

        self.actor  = MAPPOActor(obs_dim, act_dim).to(device)
        self.critic = MAPPOCritic(global_dim).to(device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self._reset_buffer()

    def _reset_buffer(self):
        self.buf = dict(obs=[], global_s=[], actions=[], rewards=[],
                        dones=[], log_probs=[], values=[])

    def select_actions(self, obs_np):
        """Sample actions for all agents from the current policy."""
        obs_t = torch.FloatTensor(obs_np).to(self.device)
        with torch.no_grad():
            dist      = self.actor.get_dist(obs_t)
            actions   = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_value(self, global_np):
        """Estimate value of the global state."""
        g = torch.FloatTensor(global_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.critic(g).item()

    def store(self, obs, global_s, actions, rewards,
              done, log_probs, value):
        """Store one transition in the rollout buffer."""
        self.buf["obs"].append(obs)
        self.buf["global_s"].append(global_s)
        self.buf["actions"].append(actions)
        self.buf["rewards"].append(rewards)
        self.buf["dones"].append(done)
        self.buf["log_probs"].append(log_probs)
        self.buf["values"].append(value)

    def compute_gae(self, last_value):
        """Compute Generalised Advantage Estimation over the stored rollout."""
        rewards = self.buf["rewards"]
        values  = self.buf["values"] + [last_value]
        dones   = self.buf["dones"]
        T       = len(rewards)

        advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        returns    = np.zeros((T, self.n_agents), dtype=np.float32)
        gae        = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            mask    = 0.0 if dones[t] else 1.0
            delta   = (np.array(rewards[t])
                       + self.gamma * values[t + 1] * mask
                       - values[t])
            gae     = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]

        return advantages, returns

    def update(self, last_value):
        """
        Perform n_epochs gradient updates on the stored rollout.

        Returns:
            Tuple of (mean_actor_loss, mean_critic_loss).
        """
        adv, ret = self.compute_gae(last_value)
        T        = len(self.buf["obs"])

        obs_t    = torch.FloatTensor(
            np.array(self.buf["obs"])).view(T * self.n_agents, -1).to(self.device)
        glb_t    = torch.FloatTensor(
            np.array(self.buf["global_s"])).to(self.device)
        acts_t   = torch.LongTensor(
            np.array(self.buf["actions"])).view(T * self.n_agents).to(self.device)
        old_lp_t = torch.FloatTensor(
            np.array(self.buf["log_probs"])).view(T * self.n_agents).to(self.device)
        adv_t    = torch.FloatTensor(
            adv).view(T * self.n_agents).to(self.device)
        ret_t    = torch.FloatTensor(
            ret).view(T * self.n_agents).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        a_losses = []
        c_losses = []

        for _ in range(self.n_epochs):
            dist    = self.actor.get_dist(obs_t)
            new_lp  = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()
            ratio   = torch.exp(new_lp - old_lp_t)
            surr1   = ratio * adv_t
            surr2   = torch.clamp(ratio,
                                  1 - self.clip_eps,
                                  1 + self.clip_eps) * adv_t
            a_loss  = (-torch.min(surr1, surr2).mean()
                       - self.ent_coef * entropy)

            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            glb_rep = glb_t.repeat_interleave(self.n_agents, dim=0)
            c_loss  = nn.functional.mse_loss(self.critic(glb_rep), ret_t)

            self.critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()

            a_losses.append(a_loss.item())
            c_losses.append(c_loss.item())

        self._reset_buffer()
        return float(np.mean(a_losses)), float(np.mean(c_losses))

    def save(self, path, episode_waits=None):
        """Save actor, critic weights and optional training history."""
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "episode_waits": episode_waits or [],
        }, path)

    @classmethod
    def load(cls, path, device="cpu", **kwargs):
        """
        Load a saved checkpoint and return a ready-to-use MAPPOTrainer.

        Args:
            path:   Path to the .pt checkpoint file.
            device: Torch device string.
            kwargs: Additional constructor arguments (lr, n_epochs, etc.).

        Returns:
            MAPPOTrainer instance with loaded weights.
        """
        ckpt    = torch.load(path, map_location=device, weights_only=False)
        trainer = cls(device=device, **kwargs)
        trainer.actor.load_state_dict(ckpt["actor"])
        trainer.critic.load_state_dict(ckpt["critic"])
        return trainer
