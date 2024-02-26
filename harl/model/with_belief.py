import copy
import torch
import torch.nn as nn
from harl.util.util import init, check
from harl.model.cnn import CNNBase
from harl.model.mlp import MLPBase, MLPLayer, BeliefProj
from harl.model.rnn import RNNLayer
from harl.model.act import ACTLayer
from harl.util.util import get_shape_from_obs_space


class Belief(nn.Module):
    """
    belief network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(Belief, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_recurrent_belief = args["use_recurrent_belief"]
        self.activation_func = args["activation_func"]
        self.recurrent_N = args["recurrent_N"]
        self.belief_option = args["belief_option"]
        self.hard_belief_thres = args["hard_belief_thres"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.obs_space = obs_space
    
        obs_shape = copy.deepcopy(get_shape_from_obs_space(self.obs_space))
        obs_shape[0] -= self.num_agents
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_belief:
            self.rnn = RNNLayer(self.hidden_sizes[-1], self.hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)
        
        # initialize a separate layer for belief network
        self.belief = BeliefProj(self.hidden_sizes[-1], self.num_agents, self.initialization_method, self.gain)
        
        self.to(device)

    def forward(self, obs, belief_rnn_states, masks):
        """
        Compute belief from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return belief: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        belief_rnn_states = check(belief_rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        belief_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_belief:
            belief_features, belief_rnn_states = self.rnn(
                belief_features, belief_rnn_states, masks)

        belief = self.belief(belief_features)

        if self.belief_option == 'hard':
            belief = torch.where(belief > self.hard_belief_thres, 1.0, 0.0)

        return belief, belief_rnn_states

