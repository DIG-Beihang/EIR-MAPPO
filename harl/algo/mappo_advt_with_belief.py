import numpy as np
import torch
import torch.nn as nn
from harl.util.util import get_grad_norm, check, update_linear_schedule
from harl.model.with_belief import Belief
from harl.algo.mappo_advt import MAPPOAdvt

class MAPPOAdvtBelief(MAPPOAdvt):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        """Initialize MAPPO algorithm."""
        super(MAPPOAdvtBelief, self).__init__(args, obs_space, act_space, num_agents, device)
        
        self.belief_lr = args["belief_lr"]
        self.use_recurrent_belief = args["use_recurrent_belief"]
        self.use_belief_active_masks = args["use_belief_active_masks"]
        # create actor network
        self.belief = Belief(args, self.obs_space, self.act_space, self.num_agents, self.device)
        # create belief optimizer
        self.belief_optimizer = torch.optim.Adam(self.belief.parameters(),
                                                 lr=self.belief_lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        super().lr_decay(episode, episodes)
        update_linear_schedule(self.belief_optimizer, episode, episodes, self.belief_lr)
    
    def get_belief(self, obs, rnn_states_belief, masks, active_masks=None):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        
        belief, rnn_states_belief = self.belief(obs[:, :-self.num_agents],
                                                rnn_states_belief,
                                                masks)
        return belief, rnn_states_belief

    def update_belief(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        (
            obs_batch,
            obs_next_batch,
            ground_truth_type_batch,
            rnn_states_batch,
            adv_rnn_states_batch,
            belief_rnn_states_batch,
            actions_batch,
            adv_actions_batch,
            masks_batch,
            active_masks_batch,
            adv_active_masks_batch,
            old_action_log_probs_batch,
            old_adv_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        ground_truth_type_batch = check(ground_truth_type_batch).to(**self.tpdv)

        # always soft update when updating belief
        belief, _ = self.get_belief(obs_batch, 
                                    belief_rnn_states_batch, 
                                    masks_batch)

        loss = nn.functional.binary_cross_entropy(belief, ground_truth_type_batch, reduction='none')

        if self.use_belief_active_masks:
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)
            loss = (loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            loss = loss.mean()

        self.belief_optimizer.zero_grad()

        loss.backward()

        if self.use_max_grad_norm:
            belief_grad_norm = nn.utils.clip_grad_norm_(
                self.belief.parameters(), self.max_grad_norm
            )
        else:
            belief_grad_norm = get_grad_norm(self.belief.parameters())

        self.belief_optimizer.step()

        return loss, belief_grad_norm

    def share_param_train_belief(self, actor_buffer, advantages, num_agents, state_type):
        """
        Perform a training update using minibatch GD.
        :param actor_buffer: (List[ActorBuffer]) buffer containing training data related to actor.
        :param advantages: (ndarray) advantages.
        :param num_agents: (int) number of agents.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["belief_loss"] = 0
        train_info["belief_grad_norm"] = 0

        if state_type == "EP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])
        
        # guess no need to use a separate epoch here?
        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                # if self.use_recurrent_policy_belief:
                data_generator = actor_buffer[agent_id].recurrent_generator_belief(
                    advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)

                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(15)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(15):
                        batches[i].append(sample[i])
                for i in range(14):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[14][0] is None:
                    batches[14] = None
                else:
                    batches[14] = np.concatenate(batches[14], axis=0)

                loss, belief_grad_norm = self.update_belief(tuple(batches))

                train_info["belief_loss"] += loss.item()
                train_info["belief_grad_norm"] += belief_grad_norm

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info