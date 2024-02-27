
import time
import numpy as np
import torch
from eir_mappo.common.popart import PopArt
from eir_mappo.common.actor_buffer_advt_with_belief import ActorBufferAdvtBelief
from eir_mappo.common.critic_buffer_ep import CriticBufferEP
from eir_mappo.common.critic_buffer_fp import CriticBufferFP
from eir_mappo.algo import ALGO_REGISTRY
from eir_mappo.common.v_critic import VCritic
from eir_mappo.common.fgsm import FGSM
import time
import numpy as np
import torch
from eir_mappo.util.util import _t2n
import setproctitle
from eir_mappo.util.util import make_eval_env, make_train_env, make_render_env, seed, init_device, init_dir, save_config, get_num_agents, softmax
from eir_mappo.env import LOGGER_REGISTRY

import ipdb


class OnPolicyMARunnerAdvtBelief:
    """Runner for on-policy algorithms (adv training)."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyMARunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        # TODO: unify the type of args
        # args: argparse.Namespace -> dict
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        # get practical parameters
        for k, v in algo_args["train"].items():
            self.__dict__[k] = v
        for k, v in algo_args["eval"].items():
            self.__dict__[k] = v
        for k, v in algo_args["render"].items():
            self.__dict__[k] = v
        assert algo_args["algo"]["belief"] == True, "You are selecting to use belief for defense, yet not enabling belief"
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_N = algo_args["model"]["recurrent_N"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.central_belief_option = algo_args["algo"].get("central_belief_option", 'mean')
        self.state_type = env_args.get("state_type", "EP")
        # TODO: don't use this default value
        self.share_param = algo_args["algo"].get("share_param", False)
        self.fixed_order = algo_args["algo"].get("fixed_order", False)
        # adv training
        self.adv_prob = algo_args["algo"].get("adv_prob", 0.5)  # probability of having adversary
        self.eval_critic_landscape = algo_args["algo"].get("eval_critic_landscape", False)  # probability of having adversary
        # adding adversary on observation
        self.obs_adversary = env_args.get("obs_agent_adversary", True)
        if not self.obs_adversary:
            print("belief needs environment to provide additional dimensions")
            raise NotImplementedError
        self.agent_adversary = algo_args["algo"].get("agent_adversary", 0)  # who is the adversary
        # use it if update adversary for multiple times
        self.victim_interval = algo_args["algo"].get("victim_interval", 1)
        # if self.agent_adversary<0, then we randomly assign adversary
        self.random_adversary = (self.agent_adversary < 0)
        self.episode_adversary = False  # which episode contains the adversary
        self.load_critic = algo_args["algo"].get("load_critic", False)
        self.load_adv_actor = algo_args["algo"].get("load_adv_actor", False)
        self.super_adversary = algo_args["algo"].get("super_adversary", False)  # whether the adversary has defenders' policies
        self.teacher_forcing = algo_args["algo"].get("teacher_forcing", False)
        self.adapt_adversary = algo_args["algo"].get("adapt_adversary", False)
        self.state_adversary = algo_args["algo"].get("state_adversary", False)
        self.render_mode = algo_args["render"].get("render_mode", None)
        self.save_checkpoint = False
        self.belief_prob = 1  # linear decay from 1 to 0

        self.algo_name = args.algo
        self.env_name = args.env
        # TODO: seed --> set_seed
        seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.use_render:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args.env, env_args, args.algo, args.exp_name, algo_args["seed"]["seed"])
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(str(args.algo) + "-" + str(args.env) + "-" + str(args.exp_name))

        # set the config of env
        if self.use_render:
            self.envs, self.manual_render, self.manual_expand_dims, self.manual_delay, self.env_num = make_render_env(
                args.env, algo_args["seed"]["seed"], env_args)
        else:
            if self.env_name == "toy":
                from eir_mappo.env.toy_example.toy_example import ToyExample
                self.toy_env = ToyExample(env_args)
            self.envs = make_train_env(
                args.env, algo_args["seed"]["seed"], algo_args["train"]["n_rollout_threads"], env_args)
            self.eval_envs = make_eval_env(
                args.env, algo_args["seed"]["seed"], algo_args["eval"]["n_eval_rollout_threads"], env_args) if algo_args["eval"]["use_eval"] else None
        self.num_agents = get_num_agents(args.env, env_args, self.envs)
        self.ground_truth_type = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents))  # self, belief of others

        self.adapt_adv_probs = np.zeros(self.num_agents)
        self.reward_max = 0

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        # actor
        if self.share_param:
            self.actor = []
            ac = ALGO_REGISTRY[args.algo](
                {**algo_args["model"], **algo_args["algo"]}, self.envs.observation_space[0], self.envs.action_space[0], self.num_agents, device=self.device)
            self.actor.append(ac)
            for agent_id in range(1, self.num_agents):
                assert self.envs.observation_space[agent_id] == self.envs.observation_space[
                    0], "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert self.envs.action_space[agent_id] == self.envs.action_space[
                    0], "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            print('Error: defense of non-shared algo with belief not implemented.')
            raise NotImplementedError
            self.actor = []
            for agent_id in range(self.num_agents):
                ac = ALGO_REGISTRY[args.algo](
                    {**algo_args["model"], **algo_args["algo"]}, self.envs.observation_space[agent_id], self.envs.action_space[agent_id], device=self.device)
                self.actor.append(ac)

        if self.use_render is False:
            # Buffer for rendering
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = ActorBufferAdvtBelief(
                    {**algo_args["train"], **algo_args["model"]}, self.envs.observation_space[agent_id], self.envs.action_space[agent_id], self.num_agents)
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]}, share_observation_space, device=self.device)
            if self.state_type == "EP":  # note that we change it to be FP here, since we need multi critics
                self.critic_buffer = CriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]}, share_observation_space)
            elif self.state_type == "FP":
                self.critic_buffer = CriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]}, share_observation_space, self.num_agents)
            else:
                raise NotImplementedError

            if self.use_popart is True:
                self.value_normalizer = PopArt(1, device=self.device)
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args.env](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir)

        if self.state_adversary or self.render_mode == "state":
            self.fgsm = FGSM(algo_args["algo"], self.obs_adversary, self.actor, device=self.device)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        if self.use_render is True:
            if self.render_mode == "state":
                self.render_adv_state()
            elif self.render_mode == "traitor":
                self.render_adv()
            else:
                self.render()
            return
        print("start running")
        self.warmup()  # reset

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        self.logger.init(episodes)

        self.logger.episode_init(0)

        if self.use_eval:
            self.prep_rollout()
            self.eval()
            if self.state_adversary:
                self.eval_adv_state()
                return
            else:
                if self.env_name not in ['pursuit']:
                    self.eval_adv()

        for episode in range(1, episodes + 1):
            if episode > episodes / 2:
                self.teacher_forcing = False
            else:
                self.save_checkpoint = True
                self.belief_prob = 1 -  (episode - 1) / (episodes / 2)
            self.ground_truth_type = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents))
            if self.use_linear_lr_decay:
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(episode)

            self.prep_rollout()
            if self.random_adversary:
                if self.adapt_adversary:
                    self.agent_adversary = np.random.choice(range(self.num_agents), p=softmax(-1 * self.adapt_adv_probs / 1))
                else:
                    self.agent_adversary = np.random.choice(range(self.num_agents))
            if episode % self.victim_interval == 0:  # which means some episodes are not adversary
                self.episode_adversary = (np.random.rand(self.n_rollout_threads) < self.adv_prob)
            else:
                self.episode_adversary = (np.random.rand(self.n_rollout_threads) < 2)  # all True
            
            # kepts the same size as belief for convinence
            self.ground_truth_type[self.episode_adversary, :, self.agent_adversary] = 1
            
            for step in range(self.episode_length):
                # Sample actions
                values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, \
                    adv_rnn_states, belief_rnn_states, rnn_states_critic = self.collect_adv(step)
                input_actions = actions.copy()
                input_actions[self.episode_adversary, self.agent_adversary] = adv_actions[self.episode_adversary, self.agent_adversary]
                # actions: (n_threads, n_agents, action_dim)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(input_actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = obs, share_obs, rewards, dones, infos, self.ground_truth_type, available_actions, \
                    values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
                    rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic

                self.logger.per_step(data)

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            self.prep_training()

            if episode % self.victim_interval == 0:
                if self.share_param:
                    actor_train_infos, critic_train_info = self.share_param_train()  # train adversary and victim
                else:
                    actor_train_infos, critic_train_info = self.train()
            else:
                if self.share_param:
                    actor_train_infos, critic_train_info = self.share_param_train_adv()  # train adversary only
                else:
                    actor_train_infos, critic_train_info = self.train_adv()
            # log information
            if episode % self.log_interval == 0:
                self.logger.episode_log(
                    actor_train_infos, critic_train_info, self.actor_buffer, self.critic_buffer)

            # eval
            if episode % self.eval_interval == 0:
                if self.use_eval:
                    self.prep_rollout()
                    self.eval()
                    if self.env_name not in ['pursuit']:
                        self.eval_adv()
                    if self.env_name == "toy" and "ddpg" not in self.args.algo:
                        self.eval_all()
                        self.eval_analytical()
                else:
                    self.save()

            self.after_update()

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(self.actor_buffer[agent_id].obs[step],
                                                                                  self.actor_buffer[agent_id].rnn_states[step],
                                                                                  self.actor_buffer[agent_id].masks[step],
                                                                                  self.actor_buffer[agent_id].available_actions[step] if self.actor_buffer[agent_id].available_actions is not None else None)
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        # [self.envs, agents, dim]
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(self.critic_buffer.share_obs[step],
                                                             self.critic_buffer.rnn_states_critic[step],
                                                             self.critic_buffer.masks[step])
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[step]),
                                                             np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def get_adv_actions(self, obs, action_probs):
        # (n_threads, n_agents, n_actions)
        action_max = np.expand_dims(action_probs.argmax(axis=-1), axis=-1)
        action_s2 = np.flip(action_max, axis=1)
        action_s1 = 1 - action_s2
        current_state = obs[:, :, 0:1]
        actions = np.zeros_like(action_max)
        actions[current_state == 0] = action_s1[current_state == 0]
        actions[current_state == 1] = action_s2[current_state == 1]

        return actions

    @torch.no_grad()
    def collect_adv(self, step):
        belief_collector = []
        action_collector = []
        adv_action_collector = []
        action_log_prob_collector = []
        adv_action_log_prob_collector = []
        rnn_state_collector = []
        belief_rnn_state_collector = []
        adv_rnn_state_collector = []
        for agent_id in range(self.num_agents):
            belief, belief_rnn_state = self.actor[agent_id].get_belief(self.actor_buffer[agent_id].obs[step],
                                                                       self.actor_buffer[agent_id].belief_rnn_states[step],
                                                                       self.actor_buffer[agent_id].masks[step])
            if self.teacher_forcing and np.random.rand() < self.belief_prob:
                belief = self.ground_truth_type[:, agent_id]
            self.actor_buffer[agent_id].obs[step][:, -self.num_agents:] = _t2n(belief)

            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(self.actor_buffer[agent_id].obs[step],
                                                                                  self.actor_buffer[agent_id].rnn_states[step],
                                                                                  self.actor_buffer[agent_id].masks[step],
                                                                                  self.actor_buffer[agent_id].available_actions[step] if self.actor_buffer[agent_id].available_actions is not None else None)
            belief_collector.append(_t2n(belief))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            belief_rnn_state_collector.append(_t2n(belief_rnn_state))
        
        for agent_id in range(self.num_agents):
            adv_obs = self.actor_buffer[agent_id].obs[step].copy()
            adv_obs[:, -self.num_agents:] = np.eye(self.num_agents)[agent_id]
            if self.super_adversary:
                def_act = np.concatenate([*action_collector[-self.num_agents:][:agent_id], 
                                          *action_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
            # currently we do not require adversary to have a belief. might need it if communication was added, but not for now
            adv_action, adv_action_log_prob, adv_rnn_state = self.actor[agent_id].get_adv_actions(adv_obs,
                                                                                                  self.actor_buffer[agent_id].adv_rnn_states[step],
                                                                                                  self.actor_buffer[agent_id].masks[step],
                                                                                                  self.actor_buffer[agent_id].available_actions[step] if self.actor_buffer[agent_id].available_actions is not None else None)
            adv_action_collector.append(_t2n(adv_action))
            adv_action_log_prob_collector.append(_t2n(adv_action_log_prob))
            adv_rnn_state_collector.append(_t2n(adv_rnn_state))
        # [self.envs, agents, dim]
        belief = np.array(belief_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        adv_actions = np.array(adv_action_collector).transpose(1, 0, 2)
        adv_action_log_probs = np.array(adv_action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        belief_rnn_states = np.array(belief_rnn_state_collector).transpose(1, 0, 2, 3)
        adv_rnn_states = np.array(adv_rnn_state_collector).transpose(1, 0, 2, 3)

        # TODO: active_masks???
        if self.central_belief_option == 'mean':
            belief_central = belief.mean(axis=1)
            belief_central = np.expand_dims(belief_central, axis=1).repeat(self.num_agents, axis=1)
        else:
            belief_central = belief

        # toy: FP, SMAC: FP
        if self.state_type == "EP":
            if self.teacher_forcing and np.random.rand() < self.belief_prob:
                belief_central = self.ground_truth_type
            self.critic_buffer.share_obs[step][:, -self.num_agents:] = belief_central.mean(axis=1)
            
            # need to change to be compatible to our setting?
            value, rnn_state_critic = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[step]),
                                                             np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        elif self.state_type == "FP":
            if self.teacher_forcing and np.random.rand() < self.belief_prob:
                belief_central = self.ground_truth_type
            self.critic_buffer.share_obs[step][:, :, -self.num_agents:] = belief_central

            value, rnn_state_critic = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[step]),
                                                             np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, ground_truth_type, available_actions, \
                    values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
                    rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        adv_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        belief_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        # masks use 0 to mask out threads that just finish
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        adv_active_masks = active_masks.copy()
        active_masks[self.episode_adversary, self.agent_adversary] = 0
        adv_active_masks[~self.episode_adversary] = 0
        adv_active_masks[:, np.arange(self.num_agents)!=self.agent_adversary] = 0

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array([[0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0] for info in infos])
        elif self.state_type == "FP":
            bad_masks = np.array([[[0.0] if "bad_transition" in info[agent_id].keys(
            ) and info[agent_id]['bad_transition'] == True else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(obs[:, agent_id], ground_truth_type[:, agent_id], rnn_states[:, agent_id], adv_rnn_states[:, agent_id], belief_rnn_states[:, agent_id], actions[:, agent_id], adv_actions[:, agent_id],
                                               action_log_probs[:, agent_id], adv_action_log_probs[:, agent_id], rewards[:, agent_id], masks[:, agent_id], active_masks[:, agent_id],
                                               adv_active_masks[:, agent_id], available_actions[:, agent_id] if available_actions[0] is not None else None)

        if self.state_type == "EP":
            self.critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)
        elif self.state_type == "FP":
            self.critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)

    @torch.no_grad()
    # calculate value using decentralized belief as advantage function
    def compute(self):
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[-1]),
                                                   np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                                                   np.concatenate(self.critic_buffer.masks[-1]))
            next_value = np.array(
                np.split(_t2n(next_value), self.n_rollout_threads))
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[-1]),
                                                   np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                                                   np.concatenate(self.critic_buffer.masks[-1]))
            next_value = np.array(
                np.split(_t2n(next_value), self.n_rollout_threads))
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        print('normal training not written for now')
        raise NotImplementedError
        actor_train_infos = []

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - \
                self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [self.actor_buffer[i].active_masks for i in range(self.num_agents)]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for agent_id in range(self.num_agents):
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP")
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP")
            actor_train_infos.append(actor_train_info)

        critic_train_info = self.critic.train(
            self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def train_adv(self):
        actor_train_infos = []

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - \
                self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [self.actor_buffer[i].active_masks for i in range(self.num_agents)]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for agent_id in range(self.num_agents):
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train_adv(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP")
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train_adv(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP")
            actor_train_infos.append(actor_train_info)

        return actor_train_infos, {}

    def share_param_train(self):
        """
        Training procedure for parameter-sharing MAPPO.
        """

        actor_train_infos = []

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - \
                self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [self.actor_buffer[i].active_masks for i in range(self.num_agents)]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        actor_train_info = self.actor[0].share_param_train(
            self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)

        if self.eval_critic_landscape:
            self.critic.visualize_critic_value_landscape(self.critic_buffer, self.value_normalizer)

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        
        if self.algo_name == "mappo_advt_belief":
            actor_train_info_belief = self.actor[0].share_param_train_belief(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)
            actor_train_info.update(actor_train_info_belief)
 
        for agent_id in torch.randperm(self.num_agents):
            actor_train_infos.append(actor_train_info)

        return actor_train_infos, critic_train_info

    def share_param_train_adv(self):
        """
        Training procedure for parameter-sharing MAPPO.
        """
        actor_train_infos = []

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - \
                self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [self.actor_buffer[i].active_masks for i in range(self.num_agents)]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        actor_train_info = self.actor[0].share_param_train_adv(
            self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)
        
        if self.algo_name == "mappo_advt_belief":
            actor_train_info_belief = self.actor[0].share_param_train_belief(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)
            actor_train_info.update(actor_train_info_belief)
        
        for agent_id in torch.randperm(self.num_agents):
            actor_train_infos.append(actor_train_info)

        return actor_train_infos, {}

    def after_update(self):
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents))

        while True:
            eval_actions_collector = []
            eval_belief_collector = []
            
            for agent_id in range(self.num_agents):
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                             eval_rnn_states_belief[:, agent_id],
                                             eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                eval_belief_collector.append(_t2n(eval_belief))

                if self.teacher_forcing and np.random.rand() < self.belief_prob:
                    eval_belief = ground_truth_type[:, agent_id]
                eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(eval_obs[:, agent_id],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                save_model = self.logger.eval_log(eval_episode)
                if save_model and self.env_name != "toy":
                    self.save()
                break

    @torch.no_grad()
    def eval_adv_state(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._eval_adv_state(i)
        else:
            return self._eval_adv_state(self.agent_adversary)

    @torch.no_grad()
    def _eval_adv_state(self, adv_id):
        self.logger.eval_init()
        eval_episode = 0
        action_changes = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
        
        while True:
            ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents))
            ground_truth_type[:, :, adv_id] = 1

            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                obs = eval_obs[:, agent_id]
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(obs,
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                
                if self.teacher_forcing and np.random.rand() < self.belief_prob:
                    eval_belief = ground_truth_type[:, agent_id]
                obs[:, -self.num_agents:] = _t2n(eval_belief)

                if agent_id == adv_id:
                    clean_obs = obs.copy()
                    obs = self.fgsm(obs,
                                    eval_rnn_states[:, agent_id],
                                    eval_adv_rnn_states[:, agent_id], 
                                    eval_masks[:, agent_id],
                                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                    agent_id=agent_id)
                else:
                    obs = eval_obs[:, agent_id]
                    clean_obs = eval_obs[:, agent_id]

                _, eval_clean_action_probs, _ = \
                    self.actor[agent_id].act_with_probs(clean_obs,
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)

                eval_actions, eval_action_probs, temp_rnn_state = \
                    self.actor[agent_id].act_with_probs(obs,
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)
                    
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

                if agent_id == adv_id:
                    action_changes.append(torch.norm(eval_clean_action_probs - eval_action_probs, p=1).item() / np.prod(eval_action_probs.shape))

            for agent_id in range(self.num_agents):
                adv_obs = eval_obs[:, agent_id].copy()
                adv_obs[:, -self.num_agents:] = np.eye(self.num_agents)[agent_id]
                if self.super_adversary:
                    def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id], 
                                              *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(adv_obs, # [:, :-self.num_agents],
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                 deterministic=False)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_adv(eval_episode, adv_id)
                break

    @torch.no_grad()
    def eval_all(self):
        self.logger.eval_init()

        eval_obs = self.toy_env.get_all_states()
        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        ground_truth_type = np.zeros((eval_obs.shape[0], self.num_agents, self.num_agents))

        eval_action_probs_collector = []
        for agent_id in range(self.num_agents):
            eval_belief, temp_rnn_state_belief = \
                self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
            eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)

            if self.teacher_forcing and np.random.rand() < self.belief_prob:
                eval_belief = ground_truth_type[:, agent_id]
            eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)

            _, eval_action_probs, _ = \
                self.actor[agent_id].act_with_probs(eval_obs[:, agent_id],
                                                    eval_rnn_states[:, agent_id],
                                                    eval_masks[:, agent_id],
                                                    None, deterministic=False)
            eval_action_probs_collector.append(_t2n(eval_action_probs))

        eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)

        self.logger.eval_log_actions(eval_action_probs)

    @torch.no_grad()
    def eval_adv(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._eval_adv(i)
        else:
            self._eval_adv(self.agent_adversary)

    @torch.no_grad()
    def _eval_adv(self, adv_id):
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        if self.obs_adversary:
            adv_len = eval_obs.shape[2] - self.num_agents + adv_id
            eval_obs[:, :, adv_len] = 1
        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        obs_offset = self.algo_args['algo'].get('obs_offset', 0)
        eval_transfer_obs = np.zeros((eval_obs.shape[0], self.num_agents, eval_obs.shape[2] + obs_offset))
        if obs_offset >= 0:
            eval_transfer_obs[:, :, :eval_obs.shape[2]] = eval_obs
        else:
            eval_transfer_obs = eval_obs[:, :, :eval_transfer_obs.shape[2]]
        if obs_offset > 0:
            transfer_adv_len = eval_obs.shape[2] - self.num_agents + adv_id
            eval_transfer_obs[:, :, transfer_adv_len] = 1
        
        while True:
            ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents))
            ground_truth_type[:, :, adv_id] = 1

            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                
                if self.teacher_forcing and np.random.rand() < self.belief_prob:
                    eval_belief = ground_truth_type[:, agent_id]
                eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(eval_obs[:, agent_id],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)
                    
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                # adv_obs = eval_obs[:, agent_id].copy()
                adv_obs = eval_transfer_obs[:, agent_id].copy()
                # adv_obs[:, -self.num_agents:] = np.eye(self.num_agents)[agent_id]
                if self.super_adversary:
                    def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id], 
                                              *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(adv_obs,
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                 deterministic=False)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
            eval_actions[:, adv_id] = eval_adv_actions[:, adv_id]

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            if self.obs_adversary:
                adv_len = eval_obs.shape[2] - self.num_agents + adv_id
                eval_obs[:, :, adv_len] = 1
            if obs_offset >= 0:
                eval_transfer_obs[:, :, :eval_obs.shape[2]] = eval_obs
            else:
                eval_transfer_obs = eval_obs[:, :, :eval_transfer_obs.shape[2]]
            if obs_offset > 0:
                transfer_adv_len = eval_obs.shape[2] - self.num_agents + adv_id
                eval_transfer_obs[:, :, transfer_adv_len] = 1


            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_adv(eval_episode, adv_id)
                break

        self.adapt_adv_probs[adv_id] = np.mean(self.logger.eval_episode_rewards)

    @torch.no_grad()
    def eval_analytical(self):
        reward = 0
        if self.random_adversary:
            for i in range(self.num_agents):
                reward += self._eval_analytical(i)
            if not self.save_checkpoint:
                self.save()
            elif reward > self.reward_max:
                self.reward_max = reward
                self.save()
        else:
            self._eval_analytical(self.agent_adversary)
            self.save()

    @torch.no_grad()
    def _eval_analytical(self, adv_id):
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents))
            ground_truth_type[:, :, adv_id] = 1
            
            eval_actions_collector = []
            eval_action_probs_collector = []
            for agent_id in range(self.num_agents):
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                
                if self.teacher_forcing and np.random.rand() < self.belief_prob:
                    eval_belief = ground_truth_type[:, agent_id]
                eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)
                
                eval_actions, eval_action_probs, temp_rnn_state = \
                    self.actor[agent_id].act_with_probs(eval_obs[:, agent_id],
                                                        eval_rnn_states[:, agent_id],
                                                        eval_masks[:, agent_id],
                                                        eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                        deterministic=False)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_action_probs_collector.append(_t2n(eval_action_probs))
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)

            adv_actions = self.get_adv_actions(eval_obs, eval_action_probs)
            eval_actions[:, adv_id] = adv_actions[:, adv_id]

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                self.logger.eval_log_analytical(eval_episode, adv_id)
                break
    
        self.adapt_adv_probs[adv_id] = np.mean(self.logger.eval_episode_rewards)
        return np.mean(self.logger.eval_episode_rewards)

    @torch.no_grad()
    def render(self):
        print("start rendering")
        obs_traj = []
        action_traj = []
        action_prob_traj = []
        belief_traj = []
        done_traj = []

        features = []
        hooks = []
        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())

        if "ddpg" in self.args.algo:
            for ii in range(self.num_agents):
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[1].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[3].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[5].register_forward_hook(hook_fn))
        else:
            hooks.append(self.actor[0].actor.base.mlp.fc[2].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[5].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[8].register_forward_hook(hook_fn))

        if self.manual_expand_dims:
            for _ in range(self.render_episodes):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                eval_rnn_states = np.zeros((self.env_num, self.num_agents,
                                            self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_rnn_states_belief = np.zeros((self.env_num, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)
                rewards = 0
                while True:
                    eval_actions_collector = []
                    eval_action_probs_collector = []
                    eval_belief_collector = []
                    for agent_id in range(self.num_agents):
                        eval_belief, temp_rnn_state_belief = \
                                            self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                        eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                        eval_belief_collector.append(_t2n(eval_belief))

                        eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)
                        eval_actions, eval_action_probs, temp_rnn_state = \
                            self.actor[agent_id].act_with_probs(eval_obs[:, agent_id],
                                                                eval_rnn_states[:, agent_id],
                                                                eval_masks[:, agent_id],
                                                                eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                                deterministic=False)
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                        eval_action_probs_collector.append(_t2n(eval_action_probs))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)
                    eval_belief = np.array(eval_belief_collector).transpose(1, 0, 2)

                    obs_traj.append(eval_obs)
                    action_traj.append(eval_actions)
                    action_prob_traj.append(eval_action_probs)
                    belief_traj.append(eval_belief)


                    # Obser reward and next obs
                    eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(
                        eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.expand_dims(np.array(
                        eval_available_actions), axis=0) if eval_available_actions is not None else None
                    if self.manual_render:
                        if "smac" not in self.args.env:  # replay for smac, no rendering
                            self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        done_traj.append(True)
                        print(f"total reward of this episode: {rewards}")
                        break
                    else:
                        done_traj.append(False)
            if "smac" in self.args.env:
                if 'v2' in self.args.env:
                    self.envs.env.save_replay()
                else:
                    self.envs.save_replay()

        else:
            raise NotImplementedError

        # np.save("traj/obs.npy", np.array(obs_traj))
        # np.save("traj/action.npy", np.array(action_traj))
        # np.save("traj/action_prob.npy", np.array(action_prob_traj))
        # np.save("traj/belief.npy", np.array(belief_traj))
        # np.save("traj/done.npy", np.array(done_traj))

        for hook in hooks:
            hook.remove()
        features_cat = []
        for i in range(len(features)//3):
            features_cat.append(np.concatenate([features[3*i], features[3*i+1], features[3*i+2]], axis=1))
        features_cat = np.concatenate(features_cat)

        np.save(f"traj/{self.args.exp_name}_features.npy", features_cat)

    @torch.no_grad()
    def render_adv(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._render_adv(i)
        else:
            self._render_adv(self.agent_adversary)

    @torch.no_grad()
    def _render_adv(self, adv_id):
        print("start adv rendering")
        obs_traj = []
        action_traj = []
        action_prob_traj = []
        belief_traj = []
        done_traj = []

        features = []
        hooks = []
        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())

        if "ddpg" in self.args.algo:
            for ii in range(self.num_agents):
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[1].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[3].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[5].register_forward_hook(hook_fn))
        else:
            hooks.append(self.actor[0].actor.base.mlp.fc[2].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[5].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[8].register_forward_hook(hook_fn))

        if self.manual_expand_dims:
            for _ in range(self.render_episodes):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)

                eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                eval_rnn_states = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_rnn_states_belief = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_adv_rnn_states = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)
                rewards = 0
                while True:
                    eval_actions_collector = []
                    eval_adv_actions_collector = []
                    eval_action_probs_collector = []
                    eval_belief_collector = []
                    for agent_id in range(self.num_agents):
                        eval_belief, temp_rnn_state_belief = \
                                            self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                        eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                        eval_belief_collector.append(_t2n(eval_belief))

                        eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)
                        eval_actions, eval_action_probs, temp_rnn_state = \
                            self.actor[agent_id].act_with_probs(eval_obs[:, agent_id],
                                                                eval_rnn_states[:, agent_id],
                                                                eval_masks[:, agent_id],
                                                                eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                                deterministic=False)
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                        eval_action_probs_collector.append(_t2n(eval_action_probs))

                    for agent_id in range(self.num_agents):
                        if self.super_adversary:
                            def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id], 
                                                    *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                            adv_obs = np.concatenate([eval_obs[:, agent_id], softmax(def_act)], axis=-1)
                        else:
                            adv_obs = eval_obs[:, agent_id]
                        eval_adv_actions, temp_adv_rnn_state = \
                            self.actor[agent_id].act_adv(adv_obs,
                                                    eval_adv_rnn_states[:, agent_id],
                                                    eval_masks[:, agent_id],
                                                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                    deterministic=False)
                        eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                        eval_adv_actions_collector.append(_t2n(eval_adv_actions))

                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
                    eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)
                    eval_belief = np.array(eval_belief_collector).transpose(1, 0, 2)

                    eval_actions[:, adv_id] = eval_adv_actions[:, adv_id]

                    obs_traj.append(eval_obs)
                    action_traj.append(eval_actions)
                    action_prob_traj.append(eval_action_probs)
                    belief_traj.append(eval_belief)
                    # Obser reward and next obs
                    eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(
                        eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)

                    eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                    if self.manual_render:
                        if "smac" not in self.args.env:  # replay for smac, no rendering
                            self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        done_traj.append(True)
                        print(f"total reward of this episode: {rewards}")
                        break
                    else:
                        done_traj.append(False)
            if "smac" in self.args.env:
                if 'v2' in self.args.env:
                    self.envs.env.save_replay()
                else:
                    self.envs.save_replay()
        else:
            raise NotImplementedError
        
        # np.save("traj/obs.npy", np.array(obs_traj))
        # np.save("traj/action.npy", np.array(action_traj))
        # np.save("traj/action_prob.npy", np.array(action_prob_traj))
        # np.save("traj/belief.npy", np.array(belief_traj))
        # np.save("traj/done.npy", np.array(done_traj))

        for hook in hooks:
            hook.remove()
        features_cat = []
        for i in range(len(features)//3):
            features_cat.append(np.concatenate([features[3*i], features[3*i+1], features[3*i+2]], axis=1))
        features_cat = np.concatenate(features_cat)

        np.save(f"traj/{self.args.exp_name}_adv{adv_id}_features.npy", features_cat)

    @torch.no_grad()
    def render_adv_state(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._render_adv_state(i)
        else:
            self._render_adv_state(self.agent_adversary)

    @torch.no_grad()
    def _render_adv_state(self, adv_id):
        print("start adv state rendering")
        obs_traj = []
        action_traj = []
        action_prob_traj = []
        belief_traj = []
        done_traj = []
        if self.manual_expand_dims:
            for _ in range(self.render_episodes):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                eval_rnn_states = np.zeros((self.env_num, self.num_agents,
                                            self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_rnn_states_belief = np.zeros((self.env_num, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_adv_rnn_states = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)
                rewards = 0
                while True:
                    eval_actions_collector = []
                    eval_action_probs_collector = []
                    eval_belief_collector = []
                    for agent_id in range(self.num_agents):
                        if agent_id == adv_id:
                            obs = self.fgsm(eval_obs[:, agent_id],
                                            eval_rnn_states[:, agent_id],
                                            eval_adv_rnn_states[:, agent_id], 
                                            eval_masks[:, agent_id],
                                            eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                            agent_id=agent_id)
                        else:
                            obs = eval_obs[:, agent_id]

                        eval_belief, temp_rnn_state_belief = \
                                            self.actor[agent_id].get_belief(obs,
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                        eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                        eval_belief_collector.append(_t2n(eval_belief))

                        eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)
                        eval_actions, eval_action_probs, temp_rnn_state = \
                            self.actor[agent_id].act_with_probs(obs,
                                                                eval_rnn_states[:, agent_id],
                                                                eval_masks[:, agent_id],
                                                                eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                                deterministic=False)
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                        eval_action_probs_collector.append(_t2n(eval_action_probs))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)
                    eval_belief = np.array(eval_belief_collector).transpose(1, 0, 2)

                    obs_traj.append(eval_obs)
                    action_traj.append(eval_actions)
                    action_prob_traj.append(eval_action_probs)
                    belief_traj.append(eval_belief)


                    # Obser reward and next obs
                    eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(
                        eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.expand_dims(np.array(
                        eval_available_actions), axis=0) if eval_available_actions is not None else None
                    if self.manual_render:
                        if "smac" not in self.args.env:  # replay for smac, no rendering
                            self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        done_traj.append(True)
                        print(f"total reward of this episode: {rewards}")
                        break
                    else:
                        done_traj.append(False)
            if "smac" in self.args.env:
                if 'v2' in self.args.env:
                    self.envs.env.save_replay()
                else:
                    self.envs.save_replay()

        else:
            raise NotImplementedError

        np.save("traj/obs.npy", np.array(obs_traj))
        np.save("traj/action.npy", np.array(action_traj))
        np.save("traj/action_prob.npy", np.array(action_prob_traj))
        np.save("traj/belief.npy", np.array(belief_traj))
        np.save("traj/done.npy", np.array(done_traj))

    def prep_rollout(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(policy_actor.state_dict(), str(
                self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")

            policy_belief = self.actor[agent_id].belief
            torch.save(policy_belief.state_dict(), str(
                self.save_dir) + "/actor_belief" + str(agent_id) + ".pt")

            adv_policy_actor = self.actor[agent_id].adv_actor
            torch.save(adv_policy_actor.state_dict(), str(
                self.save_dir) + "/adv_actor_agent" + str(agent_id) + ".pt")
        policy_critic = self.critic.critic
        torch.save(policy_critic.state_dict(), str(
            self.save_dir) + "/critic_agent" + ".pt")
        if self.value_normalizer is not None:
            torch.save(self.value_normalizer.state_dict(), str(
                self.save_dir) + "/value_normalizer" + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.actor[agent_id].actor.load_state_dict(
                policy_actor_state_dict)

            policy_belief_state_dict = torch.load(
                str(self.model_dir) + '/actor_belief' + str(agent_id) + '.pt')
            self.actor[agent_id].belief.load_state_dict(
                policy_belief_state_dict)
            if self.load_adv_actor:
                adv_policy_actor_state_dict = torch.load(
                    str(self.adv_model_dir) + '/adv_actor_agent' + str(agent_id) + '.pt')
                self.actor[agent_id].adv_actor.load_state_dict(
                    adv_policy_actor_state_dict)
        if not self.use_render and self.load_critic:
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/critic_agent' + '.pt')
            self.critic.critic.load_state_dict(
                policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(str(
                    self.model_dir) + "/value_normalizer" + ".pt")
                self.value_normalizer.load_state_dict(
                    value_normalizer_state_dict)

    def close(self):
        # post process
        if self.use_render:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(
                str(self.log_dir + '/summary.json'))
            self.writter.close()
            self.logger.close()
