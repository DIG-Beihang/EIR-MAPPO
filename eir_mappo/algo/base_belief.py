import torch
from eir_mappo.util.util import update_linear_schedule
from eir_mappo.model.actor_belief import ActorBelief
from eir_mappo.model.belief import Belief


class BaseBelief:
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        """
        Initialize Base class.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy_belief = args["use_recurrent_policy_belief"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]
        self.num_agents = num_agents

        self.lr = args["lr"]
        self.belief_lr = args["belief_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        # create actor network
        self.actor = ActorBelief(args, self.obs_space,
                                 self.act_space, self.num_agents, self.device)
        self.belief = Belief(args, self.obs_space,
                             self.act_space, self.num_agents, self.device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.belief_optimizer = torch.optim.Adam(self.belief.parameters(),
                                                 lr=self.belief_lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer,
                               episode, episodes, self.lr)

    def get_actions(self, obs, belief, rnn_states_actor, masks, available_actions=None,
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
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 belief,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)
        return actions, action_log_probs, rnn_states_actor

    def get_belief(self, obs, rnn_states_belief, masks, available_actions=None,
                   deterministic=False, belief_option='soft'):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        belief, rnn_states_belief = self.belief(obs,
                                                rnn_states_belief,
                                                masks,
                                                available_actions,
                                                deterministic,
                                                belief_option)
        return belief, rnn_states_belief

    def get_actions_with_probs(self, obs, belief, rnn_states_actor, masks, available_actions=None,
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
        actions, action_log_probs, action_probs, rnn_states_actor = self.actor.forward_with_probs(obs,
                                                                                                  belief,
                                                                                                  rnn_states_actor,
                                                                                                  masks,
                                                                                                  available_actions,
                                                                                                  deterministic)
        return actions, action_log_probs, action_probs, rnn_states_actor

    def evaluate_actions(self, obs, belief, rnn_states_actor, action, masks,
                         available_actions=None, active_masks=None):
        """Get action logprobs / entropy and value function predictions for actor update.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            action: (np.ndarray) actions whose log probabilites and entropy to compute.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        """

        action_log_probs, dist_entropy, action_distribution = self.actor.evaluate_actions(obs,
                                                                                          belief,
                                                                                          rnn_states_actor,
                                                                                          action,
                                                                                          masks,
                                                                                          available_actions,
                                                                                          active_masks)
        return action_log_probs, dist_entropy, action_distribution

    def act(self, obs, belief, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, belief, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def act_with_probs(self, obs, belief, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, action_probs, rnn_states_actor = self.actor.forward_with_probs(
            obs, belief, rnn_states_actor, masks, available_actions, deterministic)
        return actions, action_probs, rnn_states_actor

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        pass

    def train(self, actor_buffer, advantages):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (ActorBuffer) buffer containing training data related to actor.
            advantages: (ndarray) advantages.
        """
        pass

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()
        self.belief.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.actor.eval()
        self.belief.eval()
