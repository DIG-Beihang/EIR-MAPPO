import torch
import numpy as np
from harl.util.util import _flatten, _sa_cast
from harl.common.actor_buffer_advt import ActorBufferAdvt

class ActorBufferAdvtBelief(ActorBufferAdvt):
    """
    ActorBuffer contains data for on-policy actors.
    """
    def __init__(self, args, obs_space, act_space, num_agents):
        super(ActorBufferAdvtBelief, self).__init__(args, obs_space, act_space)
        self.num_agents = num_agents
        self.ground_truth_type = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.belief_rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

    def insert(self, obs, ground_truth_type, rnn_states, adv_rnn_states, belief_rnn_states, actions, adv_actions, action_log_probs, adv_action_log_probs, rewards, masks, active_masks=None, adv_active_masks=None, available_actions=None):
        self.ground_truth_type[self.step + 1] = ground_truth_type.copy()
        self.belief_rnn_states[self.step + 1] = belief_rnn_states.copy()
        super().insert(obs, rnn_states, adv_rnn_states, actions, adv_actions, action_log_probs, adv_action_log_probs, rewards, masks, active_masks, adv_active_masks, available_actions)

    def after_update(self):
        super().after_update()
        self.ground_truth_type[0] = self.ground_truth_type[-1].copy()
        self.belief_rnn_states[0] = self.belief_rnn_states[-1].copy()

    def recurrent_generator_belief(self, advantages, actor_num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(actor_num_mini_batch)]

        if len(self.obs.shape) > 3:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
            obs_next = self.obs[1:].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast(self.obs[:-1])
            obs_next = _sa_cast(self.obs[1:])
        
        ground_truth_type = _sa_cast(self.ground_truth_type)
        actions = _sa_cast(self.actions)
        adv_actions = _sa_cast(self.adv_actions)
        action_log_probs = _sa_cast(self.action_log_probs)
        adv_action_log_probs = _sa_cast(self.adv_action_log_probs)
        advantages = _sa_cast(advantages)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        adv_active_masks = _sa_cast(self.adv_active_masks[:-1])
        if self.factor is not None:
            factor = _sa_cast(self.factor)
        rnn_states = self.rnn_states[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        adv_rnn_states = self.adv_rnn_states[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.adv_rnn_states.shape[2:])
        belief_rnn_states = self.belief_rnn_states[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.belief_rnn_states.shape[2:])

        if self.available_actions is not None:
            available_actions = _sa_cast(self.available_actions[:-1])

        for indices in sampler:
            obs_batch = []
            obs_next_batch = []
            ground_truth_type_batch = []
            rnn_states_batch = []
            adv_rnn_states_batch = []
            belief_rnn_states_batch = []
            actions_batch = []
            adv_actions_batch = []
            available_actions_batch = []
            masks_batch = []
            active_masks_batch = []
            adv_active_masks_batch = []
            old_action_log_probs_batch = []
            old_adv_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                obs_batch.append(obs[ind:ind+data_chunk_length])
                obs_next_batch.append(obs_next[ind:ind+data_chunk_length])
                actions_batch.append(actions[ind:ind+data_chunk_length])
                ground_truth_type_batch.append(ground_truth_type[ind:ind+data_chunk_length])
                adv_actions_batch.append(adv_actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(
                    active_masks[ind:ind+data_chunk_length])
                adv_active_masks_batch.append(
                    adv_active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind:ind+data_chunk_length])
                old_adv_action_log_probs_batch.append(
                    adv_action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                adv_rnn_states_batch.append(adv_rnn_states[ind])
                belief_rnn_states_batch.append(belief_rnn_states[ind])
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind+data_chunk_length])
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            obs_next_batch = np.stack(obs_next_batch, axis=1)
            ground_truth_type_batch = np.stack(ground_truth_type_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            adv_actions_batch = np.stack(adv_actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            adv_active_masks_batch = np.stack(adv_active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            old_adv_action_log_probs_batch = np.stack(old_adv_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[2:])
            adv_rnn_states_batch = np.stack(adv_rnn_states_batch).reshape(
                N, *self.adv_rnn_states.shape[2:])
            belief_rnn_states_batch = np.stack(belief_rnn_states_batch).reshape(
                N, *self.belief_rnn_states.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            ground_truth_type_batch = _flatten(L, N, ground_truth_type_batch)
            obs_next_batch = _flatten(L, N, obs_next_batch)
            actions_batch = _flatten(L, N, actions_batch)
            adv_actions_batch = _flatten(L, N, adv_actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            adv_active_masks_batch = _flatten(L, N, adv_active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            old_adv_action_log_probs_batch = _flatten(L, N, old_adv_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            if self.factor is not None:
                yield obs_batch, obs_next_batch, ground_truth_type_batch, rnn_states_batch, adv_rnn_states_batch, belief_rnn_states_batch, actions_batch, adv_actions_batch,\
                    masks_batch, active_masks_batch, adv_active_masks_batch, old_action_log_probs_batch, old_adv_action_log_probs_batch,\
                    adv_targ, available_actions_batch, factor_batch
            else:
                yield obs_batch, obs_next_batch, ground_truth_type_batch, rnn_states_batch, adv_rnn_states_batch, belief_rnn_states_batch, actions_batch, adv_actions_batch,\
                    masks_batch, active_masks_batch, adv_active_masks_batch, old_action_log_probs_batch, old_adv_action_log_probs_batch,\
                    adv_targ, available_actions_batch
    