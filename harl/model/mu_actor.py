import torch
import torch.nn as nn
from harl.util.util import get_shape_from_obs_space
from harl.model.plain_cnn import PlainCNN
from harl.model.plain_mlp import PlainMLP

class MuActor(nn.Module):

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(obs_shape, hidden_sizes[0], activation_func)
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]

        self.action_type = action_space.__class__.__name__
        
        if action_space.__class__.__name__ == "Discrete":
            act_dim = action_space.n
            pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]
            self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
        elif action_space.__class__.__name__ == "Box":
            act_dim = action_space.shape[0]
            pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]
            self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
            low = torch.tensor(action_space.low).to(**self.tpdv)
            high = torch.tensor(action_space.high).to(**self.tpdv)
            self.scale = (high - low) / 2
            self.mean = (high + low) / 2
        
        self.to(device)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        # for discrete, return output logits; for box, return action value
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        x = self.pi(x)
        if self.action_type == "Box":
            x = self.scale * x + self.mean
        return x