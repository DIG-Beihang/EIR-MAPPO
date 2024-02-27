import numpy as np
import torch
import torch.nn as nn
from eir_mappo.util.util import get_grad_norm, check, softmax, update_linear_schedule
from eir_mappo.algo.base import Base
from eir_mappo.model.adv_actor import AdvActor


class MAPPOAdvt(Base):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        """Initialize MAPPO algorithm."""
        super(MAPPOAdvt, self).__init__(args, obs_space, act_space, device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.adv_lr = args["adv_lr"]
        self.adv_epoch = args["adv_epoch"]
        self.adv_entropy_coef = args["adv_entropy_coef"]
        self.super_adversary = args["super_adversary"]  # whether the adversary has defenders' policies
        self.belief = args["belief"]
        self.num_agents = num_agents

        # create actor network
        self.adv_actor = AdvActor(args, self.obs_space, self.act_space, self.num_agents, self.device)
        # create actor optimizer
        self.adv_actor_optimizer = torch.optim.Adam(self.adv_actor.parameters(),
                                                lr=self.adv_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        
    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        super().lr_decay(episode, episodes)
        update_linear_schedule(self.adv_actor_optimizer, episode, episodes, self.adv_lr)
        
    def get_adv_actions(self, obs, rnn_states_actor, masks, available_actions=None,
                        deterministic=False, agent_id=0):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs, rnn_states_actor = self.adv_actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic)
        return actions, action_log_probs, rnn_states_actor
    
    def get_adv_logits(self, obs, rnn_states_actor, masks, available_actions=None,
                        deterministic=False, agent_id=0):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        action_logits = self.adv_actor.get_logits(obs,
                                                rnn_states_actor,
                                                masks,
                                                available_actions,
                                                deterministic)
        return action_logits

    def act_adv(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.adv_actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        (
            obs_batch,
            obs_next_batch,
            rnn_states_batch,
            adv_rnn_states_batch,
            actions_batch,
            adv_actions_batch,
            rewards_batch,
            masks_batch,
            active_masks_batch,
            adv_active_masks_batch,
            old_action_log_probs_batch,
            old_adv_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            adv_obs_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch).sum() 
            if active_masks_batch.sum() > 0:
                policy_action_loss /= active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    

    def update_adv(self, sample):
        """Update adv_actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        (
            obs_batch,
            obs_next_batch,
            rnn_states_batch,
            adv_rnn_states_batch,
            actions_batch,
            adv_actions_batch,
            rewards_batch,
            masks_batch,
            active_masks_batch,
            adv_active_masks_batch,
            old_action_log_probs_batch,
            old_adv_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            adv_obs_batch,
        ) = sample

        old_adv_action_log_probs_batch = check(old_adv_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        adv_active_masks_batch = check(adv_active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        if self.super_adversary:
            obs_batch = adv_obs_batch

        adv_action_log_probs, adv_dist_entropy, _ = self.adv_actor.evaluate_actions(
            obs_batch,
            adv_rnn_states_batch,
            adv_actions_batch,
            masks_batch,
            available_actions_batch,
            adv_active_masks_batch,
        )
        # update adv_actor
        adv_imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(adv_action_log_probs - old_adv_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        adv_surr1 = adv_imp_weights * -adv_targ
        adv_surr2 = (
            torch.clamp(adv_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * -adv_targ
        )

        if self.use_policy_active_masks:
            adv_policy_action_loss = (
                -torch.sum(torch.min(adv_surr1, adv_surr2), dim=-1, keepdim=True)
                * adv_active_masks_batch).sum()
            if adv_active_masks_batch.sum() > 0:
                adv_policy_action_loss /= adv_active_masks_batch.sum()
        else:
            adv_policy_action_loss = -torch.sum(
                torch.min(adv_surr1, adv_surr2), dim=-1, keepdim=True
            ).mean()

        adv_policy_loss = adv_policy_action_loss

        self.adv_actor_optimizer.zero_grad()

        (adv_policy_loss - adv_dist_entropy * self.adv_entropy_coef).backward()

        if self.use_max_grad_norm:
            adv_actor_grad_norm = nn.utils.clip_grad_norm_(
                self.adv_actor.parameters(), self.max_grad_norm
            )
        else:
            adv_actor_grad_norm = get_grad_norm(self.adv_actor.parameters())

        self.adv_actor_optimizer.step()

        return adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights


    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (ActorBuffer) buffer containing training data related to actor.
            advantages: (ndarray) advantages.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["adv_policy_loss"] = 0
        train_info["adv_dist_entropy"] = 0
        train_info["adv_actor_grad_norm"] = 0
        train_info["adv_ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)
                for _ in range(self.adv_epoch):
                    adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights = self.update_adv(sample)
                    train_info["adv_policy_loss"] += adv_policy_loss.item() / self.adv_epoch
                    train_info["adv_actor_grad_norm"] += adv_actor_grad_norm / self.adv_epoch
                    train_info["adv_dist_entropy"] += adv_dist_entropy.item() / self.adv_epoch
                    train_info["adv_ratio"] += adv_imp_weights.mean() / self.adv_epoch

                train_info["policy_loss"] += policy_loss.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["dist_entropy"] += dist_entropy.item() 
                train_info["ratio"] += imp_weights.mean() 

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def train_adv(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (ActorBuffer) buffer containing training data related to actor.
            advantages: (ndarray) advantages.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["adv_policy_loss"] = 0
        train_info["adv_dist_entropy"] = 0
        train_info["adv_actor_grad_norm"] = 0
        train_info["adv_ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                for _ in range(self.adv_epoch):
                    adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights = self.update_adv(sample)
                    train_info["adv_policy_loss"] += adv_policy_loss.item() / self.adv_epoch
                    train_info["adv_actor_grad_norm"] += adv_actor_grad_norm / self.adv_epoch
                    train_info["adv_dist_entropy"] += adv_dist_entropy.item() / self.adv_epoch
                    train_info["adv_ratio"] += adv_imp_weights.mean() / self.adv_epoch
                

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """
        Perform a training update using minibatch GD.
        :param actor_buffer: (List[ActorBuffer]) buffer containing training data related to actor.
        :param advantages: (ndarray) advantages.
        :param num_agents: (int) number of agents.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info["adv_policy_loss"] = 0
        train_info["adv_dist_entropy"] = 0
        train_info["adv_actor_grad_norm"] = 0
        train_info["adv_ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])


        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(15)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(14):
                        batches[i].append(sample[i])
                for agent_id in range(num_agents):
                    def_act = np.concatenate([*batches[4][:agent_id], *batches[4][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([batches[0][agent_id], softmax(def_act)], axis=-1)
                    batches[14].append(adv_obs)
                for i in range(13):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[13][0] is None:
                    batches[13] = None
                else:
                    batches[13] = np.concatenate(batches[13], axis=0)
                batches[14] = np.concatenate(batches[14], axis=0)

                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(tuple(batches))
                for _ in range(self.adv_epoch):
                    adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights = self.update_adv(tuple(batches))
                    train_info["adv_policy_loss"] += adv_policy_loss.item() / self.adv_epoch
                    train_info["adv_actor_grad_norm"] += adv_actor_grad_norm / self.adv_epoch
                    train_info["adv_dist_entropy"] += adv_dist_entropy.item() / self.adv_epoch
                    train_info["adv_ratio"] += adv_imp_weights.mean() / self.adv_epoch

                train_info["policy_loss"] += policy_loss.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["dist_entropy"] += dist_entropy.item() 
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
    
    def share_param_train_adv(self, actor_buffer, advantages, num_agents, state_type):
        """
        Perform a training update using minibatch GD.
        :param actor_buffer: (List[ActorBuffer]) buffer containing training data related to actor.
        :param advantages: (ndarray) advantages.
        :param num_agents: (int) number of agents.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["adv_policy_loss"] = 0
        train_info["adv_dist_entropy"] = 0
        train_info["adv_actor_grad_norm"] = 0
        train_info["adv_ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(15)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(14):
                        batches[i].append(sample[i])
                ground_truth_type = np.squeeze(np.stack(batches[9], axis=0), -1).transpose(1, 0)
                for agent_id in range(num_agents):
                    if self.belief:
                        temp = batches[0][agent_id].copy()
                        # temp[:, -num_agents:] = np.eye(num_agents)[agent_id]
                        temp[:, -num_agents:] = ground_truth_type
                        batches[0][agent_id] = temp
                    def_act = np.concatenate([*batches[4][:agent_id], *batches[4][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([batches[0][agent_id], softmax(def_act)], axis=-1)
                    batches[14].append(adv_obs)
                for i in range(13):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[13][0] is None:
                    batches[13] = None
                else:
                    batches[13] = np.concatenate(batches[13], axis=0)
                batches[14] = np.concatenate(batches[14], axis=0)

                for _ in range(self.adv_epoch):
                    adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights = self.update_adv(tuple(batches))
                    train_info["adv_policy_loss"] += adv_policy_loss.item() / self.adv_epoch
                    train_info["adv_actor_grad_norm"] += adv_actor_grad_norm / self.adv_epoch
                    train_info["adv_dist_entropy"] += adv_dist_entropy.item() / self.adv_epoch
                    train_info["adv_ratio"] += adv_imp_weights.mean() / self.adv_epoch

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info