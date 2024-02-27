import torch
import torch.nn as nn
from eir_mappo.util.util import check


class FGSM(nn.Module):
    def __init__(self, args, obs_adversary, actor, device=torch.device("cpu")):
        super().__init__()
        self.args = args
        self.eps = args["eps"]
        self.iter = args["iter"]
        self.alpha = self.eps / args["alpha_rate"]
        self.obs_adversary = obs_adversary
        self.actor = actor
        self.num_agents = len(actor)
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.loss = nn.MSELoss()
        self.belief_attack = args.get("belief_attack", False)
        # self.loss = nn.CrossEntropyLoss()

    @torch.enable_grad()
    def forward(self, obs, rnn_states, adv_rnn_states, masks, available_actions, agent_id):
        obs = check(obs).to(**self.tpdv)
        input = obs.detach().clone()
        rnn_states = check(rnn_states).to(**self.tpdv)
        adv_rnn_states = check(adv_rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        target = self.actor[agent_id].get_adv_logits(obs, 
                                                    adv_rnn_states, 
                                                    masks, 
                                                    available_actions)
        # target = target.argmax(dim=-1).detach().clone()
        target = target.detach().clone()
        

        for i in range(self.iter):    
            obs = obs.requires_grad_()
            output = self.actor[agent_id].get_logits(obs, 
                                                    rnn_states, 
                                                    masks, 
                                                    available_actions,
                                                    agent_id=agent_id)
            # print(output[0])
            
            cost = -self.loss(output, target)

            grad = torch.autograd.grad(cost, obs, retain_graph=False, create_graph=False)[0]

            obs = obs.detach().clone() + self.alpha * grad.sign()
            delta = torch.clamp(obs - input, min=-self.eps, max=self.eps)
            # obs = torch.clamp(input + delta, min=-1, max=1).detach().clone()
            obs = (input + delta).detach().clone()
            # obs = input + torch.randn_like(input)

            if self.obs_adversary and not self.belief_attack:
                obs[:, -self.num_agents:] = input[:, -self.num_agents:]
        # exit()
        return obs.cpu().numpy()