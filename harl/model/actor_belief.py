import copy
import torch
import torch.nn as nn
from harl.util.util import init, check
from harl.model.cnn import CNNBase
from harl.model.mlp import MLPBase
from harl.model.rnn import RNNLayer
from harl.model.act import ACTLayer
from harl.util.util import get_shape_from_obs_space, get_obs_space_type


class ActorBelief(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(ActorBelief, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_recurrent_policy_belief = args["use_recurrent_policy_belief"]
        self.recurrent_N = args["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = copy.deepcopy(get_shape_from_obs_space(obs_space))

        # add belief
        obs_shape[0] = obs_shape[0] + num_agents

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
            self.rnn = RNNLayer(self.hidden_sizes[-1], self.hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)

        self.act = ACTLayer(action_space, self.hidden_sizes[-1],
                            self.initialization_method, self.gain, args)
    
        self.to(device)

    def forward(self, obs, belief, rnn_states, masks,  available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param masks: (np.ndarray / torch.Tensor) belief tensor denoting type of other agents. 0: cooperative 1: adversarial
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        belief = check(belief).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        obs = torch.cat([obs, belief], dim=-1)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def forward_with_probs(self, obs, belief, rnn_states, masks,  available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        belief = check(belief).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        obs = torch.cat([obs, belief], dim=-1)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic)
        action_probs = self.act.get_probs(actor_features, available_actions)

        return actions, action_log_probs, action_probs, rnn_states

    def evaluate_actions(self, obs, belief, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        belief = check(belief).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        obs = torch.cat([obs, belief], dim=-1)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(actor_features,
                                                                                                     action, available_actions,
                                                                                                     active_masks=active_masks if self.use_policy_active_masks
                                                                                                     else None)

        return action_log_probs, dist_entropy, action_distribution
