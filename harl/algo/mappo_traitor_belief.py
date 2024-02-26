import numpy as np
import torch
import torch.nn as nn
from harl.util.util import get_grad_norm, check
from harl.algo.mappo_advt_with_belief import MAPPOAdvtBelief
from harl.model.actor import Actor


class MAPPOTraitorBelief(MAPPOAdvtBelief):
    def train(self, actor_buffer, advantages, state_type):
        return self.train_adv(actor_buffer, advantages, state_type)

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        return self.share_param_train_adv(actor_buffer, advantages, num_agents, state_type)
    