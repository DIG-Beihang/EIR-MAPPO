from eir_mappo.model.distributions import Categorical, DiagGaussian
import torch.nn as nn
import torch


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param initialization_method: (str) initialization method.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, initialization_method, gain, args=None):
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim, action_dim, initialization_method, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                inputs_dim, action_dim, initialization_method, gain, args)
        else:
            action_dim = action_space[0]
            self.action_out = DiagGaussian(
                inputs_dim, action_dim, initialization_method, gain, args)

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs
    
    def forward_fakem3(self, eps, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        action_logits = self.action_out(x, available_actions)
        actions_sample = action_logits.sample()
        actions_min = action_logits.min_mode()

        randn = torch.rand_like(actions_min.float())
        actions = torch.zeros_like(actions_min)

        actions[randn>eps] = actions_sample[randn>eps]
        actions[randn<eps] = actions_min[randn<eps]
        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs

        return action_probs

    def get_logits(self, x, available_actions=None):
        """
        Compute action logits from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_logits: (torch.Tensor)
        """
        action_logits = self.action_out.transit(x, available_actions)
        return action_logits

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_distribution = self.action_out(x, available_actions)
        action_log_probs = action_distribution.log_probs(action)
        if active_masks is not None:
            if self.action_type == "Discrete":
                dist_entropy = (action_distribution.entropy(
                )*active_masks.squeeze(-1)).sum()
            else:
                dist_entropy = (action_distribution.entropy() *
                                active_masks).sum()
            if active_masks.sum() > 0:
                dist_entropy /= active_masks.sum()
        else:
            dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy, action_distribution
