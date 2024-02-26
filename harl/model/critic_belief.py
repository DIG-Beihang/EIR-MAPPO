import torch
import torch.nn as nn
from harl.util.util import init, check
from harl.model.cnn import CNNBase
from harl.model.mlp import MLPBase
from harl.model.rnn import RNNLayer
from harl.util.util import get_shape_from_obs_space, get_init_method
import copy


class CriticBelief(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, num_agents, device=torch.device("cpu")):
        super(CriticBelief, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.critic_option = args["critic_option"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_recurrent_policy_belief = args["use_recurrent_policy_belief"]
        self.recurrent_N = args["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        init_method = get_init_method(self.initialization_method)
        self.num_agents = num_agents

        cent_obs_shape = copy.deepcopy(get_shape_from_obs_space(cent_obs_space))
        cent_obs_shape[0] += num_agents
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
            self.rnn = RNNLayer(self.hidden_sizes[-1], self.hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, belief, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        belief = check(belief).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.critic_option == 'direct':
            cent_obs = torch.cat([cent_obs, belief], dim=-1)

            critic_features = self.base(cent_obs)
            
            # if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
            #     critic_features, rnn_states = self.rnn(
            #         critic_features, rnn_states, masks)
            values_final = self.v_out(critic_features)
        elif self.critic_option == 'factorize':
            batch_size = cent_obs.shape[0]
            # N+1 types
            cent_obs = cent_obs.unsqueeze(1).repeat(1, self.num_agents + 1, 1)
            type_adv = torch.eye(self.num_agents).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            type_coop = torch.zeros(batch_size, 1, self.num_agents).to(self.device)
            type_mix = torch.cat([type_coop, type_adv], dim=1)
            cent_obs = torch.cat([cent_obs, type_mix], dim=-1)

            critic_features = self.base(cent_obs)
                # being lazy here, so not to implement RNN hidden states for each belief type
                # if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_policy_belief:
                #     critic_features, rnn_states = self.rnn(
                #         critic_features, rnn_states, masks)
            values = self.v_out(critic_features)
            # v_coop - sum(b_i * softplut(v_adv_bi)), which is to ensure adv V function always have negative contribution
            values_final = values[:, 0, :] - torch.einsum('ijk, ij ->ik', torch.nn.functional.softplus(values[:, 1:, :]), belief)

        else:
            raise NotImplementedError('critic option not supported. choose between direct and factorize')

        return values_final, rnn_states
