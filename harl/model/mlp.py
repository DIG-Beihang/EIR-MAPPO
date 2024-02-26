import torch.nn as nn
from harl.util.util import init, get_active_func, get_init_method

"""MLP modules."""

# newly added, one layer linear layer
class BeliefProj(nn.Module):
    def __init__(self, input_dim, out_dim, initialization_method, gain=0.01):
        super(BeliefProj, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc = init_(nn.Linear(input_dim, out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [init_(nn.Linear(input_dim, hidden_sizes[0])), active_func, nn.LayerNorm(hidden_sizes[0])]

        for i in range(1, len(hidden_sizes)):
            layers += [init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])), active_func, nn.LayerNorm(hidden_sizes[i])]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func)

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
    

class AdvMLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(AdvMLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func)

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
