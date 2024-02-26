import torch
from copy import deepcopy
from harl.model.continuous_q_net import ContinuousQNet
import torch.nn.functional as F
from harl.util.util import check, update_linear_schedule
class ContinuousQCritic:
    def __init__(self, args, share_obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr)
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer,
                               step, steps, self.critic_lr)

    def soft_update(self):
        for param_target, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak)

    def get_values(self, share_obs, actions):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return self.critic(share_obs, actions)

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
        actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_q_values = self.target_critic(next_share_obs, next_actions)

        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)

        q_values = self.critic(share_obs, actions)
        # print(torch.stack([q_values[:, 0], q_targets[:, 0]], dim=1))
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # print(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        torch.save(self.critic.state_dict(), str(
            save_dir) + "/critic_agent" + ".pt")
        torch.save(self.target_critic.state_dict(), str(
            save_dir) + "/target_critic_agent" + ".pt")

    def restore(self, model_dir):
        critic_state_dict = torch.load(
            str(model_dir) + '/critic_agent' + '.pt')
        self.critic.load_state_dict(
            critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + '/target_critic_agent' + '.pt')
        self.target_critic.load_state_dict(
            target_critic_state_dict)

    def turn_on_grad(self):
        for p in self.critic.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        for p in self.critic.parameters():
            p.requires_grad = False
