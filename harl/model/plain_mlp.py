import torch.nn as nn
from harl.util.util import get_active_func

class PlainMLP(nn.Module):
    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes)-1):
            act = activation_func if j < len(sizes)-2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j+1]), get_active_func(act)]
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)