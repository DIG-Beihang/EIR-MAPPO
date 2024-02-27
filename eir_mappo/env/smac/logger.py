from eir_mappo.common.base_logger import BaseLogger
import numpy as np
from functools import reduce


class SMACLogger(BaseLogger):

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.best_eval_win_rate = 0

    def init(self, episodes):
        super(SMACLogger, self).init(episodes)
        self.episode_lens = []
        self.one_episode_len = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=int)
        self.last_battles_game = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32)
        self.last_battles_won = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32)

    def episode_init(self, episode):
        super(SMACLogger, self).episode_init(episode)

    def per_step(self, data):
        dones, infos = data[3], data[4]
        self.infos = infos
        self.one_episode_len += 1
        done_env = np.all(dones, axis=1)
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if done_env[i]:
                self.episode_lens.append(self.one_episode_len[i].copy())
                self.one_episode_len[i] = 0        

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        super(SMACLogger, self).episode_log(
            actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
              .format(self.env_args["map_name"],
                      self.args.algo,
                      self.args.exp_name,
                      self.episode,
                      self.episodes,
                      self.total_num_steps,
                      self.algo_args["train"]["num_env_steps"],
                      int(self.total_num_steps / (self.end - self.start))))

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []

        for i, info in enumerate(self.infos):
            if 'battles_won' in info[0].keys():
                battles_won.append(info[0]['battles_won'])
                incre_battles_won.append(
                    info[0]['battles_won']-self.last_battles_won[i])
            if 'battles_game' in info[0].keys():
                battles_game.append(info[0]['battles_game'])
                incre_battles_game.append(
                    info[0]['battles_game']-self.last_battles_game[i])

        incre_win_rate = np.sum(
            incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0
        self.writter.add_scalar(
            "env/incre_win_rate", incre_win_rate, self.total_num_steps)
        
        self.last_battles_game = battles_game
        self.last_battles_won = battles_won

        average_episode_len = np.mean(
            self.episode_lens) if len(self.episode_lens) > 0 else 0.0
        self.episode_lens = []

        self.writter.add_scalar(
            "env/average_episode_length", average_episode_len, self.total_num_steps)
        
        for agent_id in range(self.num_agents):
            actor_train_infos[agent_id]['dead_ratio'] = 1 - actor_buffer[agent_id].active_masks.sum() / (
                self.num_agents * reduce(lambda x, y: x*y, list(actor_buffer[agent_id].active_masks.shape)))
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)


        print("incre games {:.4f}, win rate on these games is {:.4f}, average step reward is {:.4f}, average episode length is {:.4f}, average episode return is {:.4f}.".format(np.sum(incre_battles_game), incre_win_rate, critic_train_info["average_step_rewards"], average_episode_len, average_episode_len * critic_train_info["average_step_rewards"]))

    def eval_init(self):
        super().eval_init()
        self.eval_battles_won = 0
        self.eval_episode_rewards_positive = []
        self.one_episode_rewards_positive = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards_positive.append([])
            self.eval_episode_rewards_positive.append([])

    def eval_per_step(self, eval_data):
        super().eval_per_step(eval_data)
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards_positive[eval_i].append(eval_infos[eval_i][0]["reward_positive"])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self.eval_episode_rewards_positive[tid].append(
            np.sum(self.one_episode_rewards_positive[tid], axis=0))
        self.one_episode_rewards_positive[tid] = []
        if self.eval_infos[tid][0]['won']:
            self.eval_battles_won += 1
        

    def eval_log(self, eval_episode):
        super().eval_log(eval_episode)
        self.eval_episode_rewards_positive = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards_positive if rewards])
        eval_win_rate = self.eval_battles_won/eval_episode
        eval_env_infos = {'eval_return_mean': self.eval_episode_rewards,
                          'eval_return_std': [np.std(self.eval_episode_rewards)],
                          'eval_return_positive_mean': self.eval_episode_rewards_positive,
                          'eval_win_rate': [eval_win_rate]}
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("eval win rate is {}, eval average episode return is {}.".format(eval_win_rate, eval_avg_rew))
        self.log_file.write(",".join(map(str, [self.total_num_steps, eval_avg_rew, eval_win_rate])) + "\n")
        self.log_file.flush()
        return True
        # if eval_win_rate > self.best_eval_win_rate:
        #     self.best_eval_win_rate = eval_win_rate
        #     return True
        # else:
        #     return False
        
    def eval_log_adv(self, eval_episode, agent_id):
        super().eval_log(eval_episode)
        self.eval_episode_rewards_positive = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards_positive if rewards])
        eval_win_rate = self.eval_battles_won/eval_episode
        eval_env_infos = {'eval_adv{}_return_mean'.format(agent_id): self.eval_episode_rewards,
                          'eval_adv{}_return_std'.format(agent_id): [np.std(self.eval_episode_rewards)],
                          'eval_adv{}_return_positive'.format(agent_id): self.eval_episode_rewards_positive,
                          'eval_adv{}_win_rate'.format(agent_id): [eval_win_rate]}
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("eval adv{} win rate is {}, eval adv average episode return is {}.".format(agent_id, eval_win_rate, eval_avg_rew))
        return eval_avg_rew
