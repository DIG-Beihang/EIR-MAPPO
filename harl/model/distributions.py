import torch
import torch.nn as nn
from harl.util.util import init
from harl.util.util import get_init_method

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
    
    def min_mode(self):
        probs_clone = self.probs.clone()
        probs_clone[probs_clone < 1e-5] = 100
        return probs_clone.argmin(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean
    
    def min_mode(self):
        return torch.rand_like(self.mean)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)

    def transit(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return x

# class DiagGaussian(nn.Module):
#     def __init__(self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01):
#         super(DiagGaussian, self).__init__()
#
#         init_method = get_init_method(initialization_method)
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
#
#         self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         self.logstd = AddBias(torch.zeros(num_outputs))
#
#     def forward(self, x, available_actions=None):
#         action_mean = self.fc_mean(x)
#
#         #  An ugly hack for my KFAC implementation.
#         zeros = torch.zeros(action_mean.size())
#         if x.is_cuda:
#             zeros = zeros.cuda()
#
#         action_logstd = self.logstd(zeros)
#         return FixedNormal(action_mean, action_logstd.exp())


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01, args=None):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(
            self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
    
    def transit(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        return action_mean


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
