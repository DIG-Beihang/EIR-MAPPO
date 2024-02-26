import torch
import torch.nn as nn
from harl.util.util import init, check
from harl.model.cnn import CNNBase
from harl.model.mlp import MLPBase
from harl.model.rnn import RNNLayer
from harl.util.util import get_shape_from_obs_space


class VanillaQNet(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(VanillaQNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_N = args["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_sizes[-1], self.hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)

        self.act = nn.Linear(self.hidden_sizes[-1], action_space.n)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states, masks)

        action_qs = self.act(actor_features)
        if available_actions is not None:
            action_qs[available_actions==0] = -1e10

        return action_qs, rnn_states
