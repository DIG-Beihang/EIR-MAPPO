import torch
from copy import deepcopy
from harl.model.continuous_q_net import ContinuousQNet
import torch.nn.functional as F
from harl.util.util import check, update_linear_schedule
from harl.common.continuous_q_critic import ContinuousQCritic

class ContinuousQCriticNS(ContinuousQCritic):
    def __init__(self, args, share_obs_space, act_space, device=torch.device("cpu")):
        super(ContinuousQCriticNS, self).__init__(args, share_obs_space, act_space, device)
        self.epsilon = args["epsilon"]
    
    def get_noise(self, share_obs, actions):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        n_agents, batch_size, dim = actions.shape
        # (n_agents, batch_size, dim) --> (batch_size, n_agents * dim)
        actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)

        actions.requires_grad_(True)
        q_values = self.critic(share_obs, actions)
        q_values.sum().backward()
        noise = actions.grad.data.detach().clone()
        
        noise = noise.view(batch_size, n_agents, dim).transpose(0, 1)

        return self.epsilon * noise.cpu().numpy()
    
    def get_target_noise(self, share_obs, actions):
        actions = actions.detach().clone()
        actions.requires_grad_(True)
        q_values = self.target_critic(share_obs, actions)
        q_values.sum().backward()
        noise = actions.grad.data.detach().clone()

        return self.epsilon * noise

    def train(self, share_obs, actions, reward, done, term, next_share_obs, next_actions, gamma):
        """
        update model.
        share_obs: (batch_size, dim)
        actions: (n_agents, batch_size, dim)
        reward: (batch_size, 1)
        done: (batch_size, 1)
        term: (batch_size, 1)
        next_share_obs: (batch_size, dim)
        next_actions: (n_agents, batch_size, dim)
        gamma: (batch_size, 1)
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        n_agents, batch_size, dim = actions.shape
        actions = torch.cat([actions[i] for i in range(n_agents)], dim=-1)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        noise = self.get_target_noise(next_share_obs, next_actions)

        critic_loss = 0
        for agent_id in range(n_agents):
            masks = torch.ones_like(noise)
            masks[:, agent_id * dim: (agent_id + 1) * dim] = 0
            next_noise_actions = next_actions - noise * masks
            next_q_values = self.target_critic(next_share_obs, next_noise_actions)
            if self.use_proper_time_limits:
                q_targets = reward + gamma * next_q_values * (1 - term)
            else:
                q_targets = reward + gamma * next_q_values * (1 - done)
            critic_loss += torch.mean(F.mse_loss(self.critic(share_obs, actions), q_targets))

        self.critic_optimizer.zero_grad()
        (critic_loss / n_agents).backward()
        self.critic_optimizer.step()
