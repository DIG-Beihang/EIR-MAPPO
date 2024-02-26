import torch
from copy import deepcopy
from harl.model.dueling_q_net import DuelingQNet
import torch.nn.functional as F
from harl.util.util import check, update_linear_schedule
import copy


class DiscreteQCritic:
    def __init__(self, args, share_obs_space, action_spaces, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.process_action_spaces(action_spaces)
        self.critic = DuelingQNet(
            args, share_obs_space, self.joint_action_dim, device)
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
        actions = check(actions).to(**self.tpdv_a)
        joint_action = self.distr_to_joint(actions)
        return torch.gather(self.critic(share_obs), 1, joint_action)

    def train_values(self, share_obs, actions):
        """"
        get target values for actor update.
        params:
        share_obs: (batch_size, dim)
        actions: [(batch_size, 1)]
        returns:
        update_actions: inner func for updating actions
        get_values: inner func for getting values
        """
        share_obs = check(share_obs).to(**self.tpdv)
        all_values = self.critic(share_obs)
        actions = copy.deepcopy(actions)

        def update_actions(agent_id):
            joint_idx = self.get_joint_idx(actions, agent_id)
            values = torch.gather(all_values, 1, joint_idx)
            action = torch.argmax(values, dim=-1, keepdim=True)
            actions[agent_id] = action

        def get_values():
            joint_action = self.distr_to_joint(actions)
            return torch.gather(all_values, 1, joint_action)

        return update_actions, get_values

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
        actions = check(actions).to(**self.tpdv_a)
        action = self.distr_to_joint(actions).to(**self.tpdv_a)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_action = self.distr_to_joint(next_actions).to(**self.tpdv_a)
        next_q_values = torch.gather(
            self.target_critic(next_share_obs), 1, next_action)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss = torch.mean(F.mse_loss(
            torch.gather(self.critic(share_obs), 1, action), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def process_action_spaces(self, action_spaces):
        self.action_dims = []
        self.joint_action_dim = 1
        for space in action_spaces:
            self.action_dims.append(space.n)
            self.joint_action_dim *= space.n

    def joint_to_distr(self, orig_action):
        """
        Convert joint action to distributed actions.
        For example, if agents' action_dims are [4, 3],
        then: 
        joint action 0 <--> distr actions [0, 0],
        joint action 1 <--> distr actions [1, 0],
        ......
        joint action 5 <--> distr actions [1, 1],
        ......
        joint action 11 <--> distr actions [3, 2].

        params:
        orig_action: torch tensor, tpdv.
        returns:
        actions: [torch tensor], tpdv.
        """
        action = copy.deepcopy(orig_action)
        actions = []
        for dim in self.action_dims:
            actions.append(action % dim)
            action = torch.div(action, dim, rounding_mode="floor")
        return actions

    def distr_to_joint(self, orig_actions):
        """
        Convert distributed actions to joint action.
        For example, if agents' action_dims are [4, 3],
        then: 
        joint action 0 <--> distr actions [0, 0],
        joint action 1 <--> distr actions [0, 1],
        ......
        joint action 5 <--> distr actions [1, 2],
        ......
        joint action 11 <--> distr actions [3, 2].

        params:
        orig_actions: [torch tensor], tpdv.
        returns:
        action: torch tensor, tpdv.
        """
        actions = copy.deepcopy(orig_actions)
        action = torch.zeros_like(actions[0])
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            action += accum_dim * actions[i]
            accum_dim *= dim
        return action

    def get_joint_idx(self, actions, agent_id):
        # joint_idx = self.get_joint_idx(actions, agent_id)
        """
        Get available joint idx for an agent.
        All other agents keep their current actions,
        and this agent can freely choose.
        params:
        actions: [(batch_size, 1)]
        agent_id: int
        returns:
        joint_idx: (batch_size, self.action_dims[agent_id]) torch.tensor, tpdv
        """
        batch_size = actions[0].shape[0]
        joint_idx = torch.zeros(
            (batch_size, self.action_dims[agent_id])).to(**self.tpdv_a)
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            if i == agent_id:
                for j in range(self.action_dims[agent_id]):
                    joint_idx[:, j] += accum_dim * j
            else:
                joint_idx += accum_dim * actions[i]
            accum_dim *= dim
        return joint_idx

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
