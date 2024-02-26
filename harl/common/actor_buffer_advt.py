import torch
import numpy as np
from collections import defaultdict
from harl.util.util import check, get_shape_from_obs_space, get_shape_from_act_space, _flatten, _sa_cast

class ActorBufferAdvt:
    """
    ActorBuffer contains data for on-policy actors.
    """
    def __init__(self, args, obs_space, act_space):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_N = args["recurrent_N"]

        obs_shape = get_shape_from_obs_space(obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.adv_rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.adv_actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.adv_action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)
        self.adv_active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def insert(self, obs, rnn_states, adv_rnn_states, actions, adv_actions, action_log_probs, adv_action_log_probs, rewards, masks, active_masks=None, adv_active_masks=None, available_actions=None):
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.adv_rnn_states[self.step + 1] = adv_rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.adv_actions[self.step] = adv_actions.copy()
        self.adv_action_log_probs[self.step] = adv_action_log_probs.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if adv_active_masks is not None:
            self.adv_active_masks[self.step + 1] = adv_active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.adv_rnn_states[0] = self.adv_rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.adv_active_masks[0] = self.adv_active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def feed_forward_generator_actor(self, advantages, actor_num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          actor_num_mini_batch))
            mini_batch_size = batch_size // actor_num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(actor_num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs_next = self.obs[1:].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        adv_rnn_states = self.adv_rnn_states[:-1].reshape(-1, *self.adv_rnn_states.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        adv_actions = self.adv_actions.reshape(-1, self.adv_actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        rewards = self.rewards.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        adv_active_masks = self.adv_active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1])
        adv_action_log_probs = self.adv_action_log_probs.reshape(
            -1, self.adv_action_log_probs.shape[-1])
        if self.factor is not None:
            # factor = self.factor.reshape(-1,1)
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            obs_next_batch = obs_next[indices]
            rnn_states_batch = rnn_states[indices]
            adv_rnn_states_batch = adv_rnn_states[indices]
            actions_batch = actions[indices]
            adv_actions_batch = adv_actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            rewards_batch = rewards[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            adv_active_masks_batch = adv_active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            old_adv_action_log_probs_batch = adv_action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.factor is None:
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch
            else:
                factor_batch = factor[indices]
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch

    def naive_recurrent_generator_actor(self, advantages, actor_num_mini_batch):
        raise NotImplementedError
        n_rollout_threads = self.actions.shape[1]
        assert n_rollout_threads >= actor_num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, actor_num_mini_batch))
        num_envs_per_batch = n_rollout_threads // actor_num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        self.available_actions[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.factor is not None:
                    factor_batch.append(self.factor[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(
                old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(
                N, *self.rnn_states.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(
                    T, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(T, N, factor_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(
                T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)
            if self.factor is not None:
                yield obs_batch, rnn_states_batch, actions_batch, \
                    masks_batch, active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch, factor_batch
            else:
                yield obs_batch, rnn_states_batch, actions_batch, \
                    masks_batch, active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length):
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
            obs = self.obs[:-1].transpose(1, 0, 2,
                                          3, 4).reshape(-1, *self.obs.shape[2:])
            obs_next = self.obs[1:].transpose(1, 0, 2,
                                          3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast(self.obs[:-1])
            obs_next = _sa_cast(self.obs[1:])

        actions = _sa_cast(self.actions)
        adv_actions = _sa_cast(self.adv_actions)
        action_log_probs = _sa_cast(self.action_log_probs)
        adv_action_log_probs = _sa_cast(self.adv_action_log_probs)
        advantages = _sa_cast(advantages)
        rewards = _sa_cast(self.rewards)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        adv_active_masks = _sa_cast(self.adv_active_masks[:-1])
        if self.factor is not None:
            factor = _sa_cast(self.factor)
        rnn_states = self.rnn_states[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        adv_rnn_states = self.adv_rnn_states[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.adv_rnn_states.shape[2:])

        if self.available_actions is not None:
            available_actions = _sa_cast(self.available_actions[:-1])

        for indices in sampler:
            obs_batch = []
            obs_next_batch = []
            rnn_states_batch = []
            adv_rnn_states_batch = []
            actions_batch = []
            adv_actions_batch = []
            available_actions_batch = []
            rewards_batch = []
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
                adv_actions_batch.append(adv_actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind:ind+data_chunk_length])
                rewards_batch.append(rewards[ind:ind+data_chunk_length])
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
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind+data_chunk_length])
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            obs_next_batch = np.stack(obs_next_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            adv_actions_batch = np.stack(adv_actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
            rewards_batch = np.stack(rewards_batch, axis=1)
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

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            obs_next_batch = _flatten(L, N, obs_next_batch)
            actions_batch = _flatten(L, N, actions_batch)
            adv_actions_batch = _flatten(L, N, adv_actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(
                    L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            rewards_batch = _flatten(L, N, rewards_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            adv_active_masks_batch = _flatten(L, N, adv_active_masks_batch)
            old_action_log_probs_batch = _flatten(
                L, N, old_action_log_probs_batch)
            old_adv_action_log_probs_batch = _flatten(
                L, N, old_adv_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            if self.factor is not None:
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, \
                    adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, \
                    old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield obs_batch, obs_next_batch, rnn_states_batch, adv_rnn_states_batch, actions_batch, \
                    adv_actions_batch, rewards_batch, masks_batch, active_masks_batch, adv_active_masks_batch, \
                    old_action_log_probs_batch, old_adv_action_log_probs_batch, adv_targ, available_actions_batch
