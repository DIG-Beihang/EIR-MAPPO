import copy
import torch
import torch.nn as nn
from harl.util.util import init, check
from harl.model.cnn import CNNBase
from harl.model.mlp import MLPBase
from harl.model.rnn import RNNLayer
from harl.model.act import ACTLayer
from harl.model.actor import Actor
from harl.util.util import get_shape_from_obs_space, get_shape_from_act_space


class AdvActor(Actor):
    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        args = copy.deepcopy(args)
        if "adv_hidden_sizes" in args:
            args['hidden_sizes'] = args["adv_hidden_sizes"]
        super(AdvActor, self).__init__(args, obs_space, action_space, device)
        self.super_adversary = args["super_adversary"]  # whether the adversary has defenders' policies
        self.adv_hidden_sizes = args["hidden_sizes"]
        # self.adv_hidden_sizes = args["adv_hidden_sizes"]
        # print(self.adv_hidden_sizes)
        # exit(0)
        self.obs_offset = args.get("obs_offset", 0)
        obs_shape = copy.deepcopy(get_shape_from_obs_space(obs_space))

        if self.super_adversary:
            obs_shape[0] = obs_shape[0] + (num_agents - 1) * get_shape_from_act_space(action_space)
        obs_shape[0] = obs_shape[0] + self.obs_offset
        self.obs_len = obs_shape[0]

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(self.adv_hidden_sizes[-1], self.adv_hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)

        self.act = ACTLayer(action_space, self.adv_hidden_sizes[-1],
                            self.initialization_method, self.gain, args)

        self.to(device)

    # def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
    #     """
    #     Compute actions from the given inputs.
    #     :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
    #     :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
    #     :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
    #     :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
    #                                                           (if None, all actions available)
    #     :param deterministic: (bool) whether to sample from action distribution or return the mode.

    #     :return actions: (torch.Tensor) actions to take.
    #     :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
    #     :return rnn_states: (torch.Tensor) updated RNN hidden states.
    #     """
    #     obs = check(obs).to(**self.tpdv)
    #     rnn_states = check(rnn_states).to(**self.tpdv)
    #     masks = check(masks).to(**self.tpdv)
    #     if available_actions is not None:
    #         available_actions = check(available_actions).to(**self.tpdv)
    #     if obs.shape[-1] > self.obs_len:
    #         obs = obs[:, :self.obs_len]
    #     else:
    #         a = torch.zeros((obs.shape[0], self.obs_len - obs.shape[-1])).to(**self.tpdv)
    #         obs = torch.cat([obs, a], dim=-1)
    #     actor_features = self.base(obs)

    #     if self.use_naive_recurrent_policy or self.use_recurrent_policy:
    #         actor_features, rnn_states = self.rnn(
    #             actor_features, rnn_states, masks)

    #     actions, action_log_probs = self.act(
    #         actor_features, available_actions, deterministic)

    #     return actions, action_log_probs, rnn_states
