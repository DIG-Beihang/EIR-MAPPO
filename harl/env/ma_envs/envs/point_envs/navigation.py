import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from harl.env.ma_envs.commons.utils import EzPickle
from harl.env.ma_envs import base
# from ma_envs.envs.environment import MultiAgentEnv
from harl.env.ma_envs.agents.point_agents.navigate_agent import NavigateAgent
from harl.env.ma_envs.commons import utils as U
import networkx as nwx
import itertools
import matplotlib.pyplot as plt
try:
    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpla
    from matplotlib.patches import Wedge
    from matplotlib.patches import RegularPolygon
    import matplotlib.patches as patches
except:
    pass


class NavigationEnv_ori(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self,
                 windows_size=1,
                 use_history=False,
                 nr_pursuers=5,
                 int_points_num=1,
                 obs_mode='2D_rbf',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=8,
                 bearing_bins=8,
                 torus=True,
                 dynamics='direct'):
        EzPickle.__init__(self, nr_pursuers, int_points_num, obs_mode, comm_radius, world_size, distance_bins,
                          bearing_bins, torus, dynamics)
        self.nr_agents = nr_pursuers
        self.n_agents = self.nr_agents
        self.int_points_num = int_points_num
        self.obs_mode = obs_mode
        self.distance_bins = distance_bins
        self.bearing_bins = bearing_bins
        self.comm_radius = comm_radius
        self.obs_radius = comm_radius / 2
        self.torus = torus
        self.dynamics = dynamics
        self.world_size = world_size
        self.world = base.World(world_size, torus, dynamics)
        self.world.agents = [
            NavigateAgent(self) for _ in
            range(self.nr_agents)
        ]
        self.world.int_points = []
        # 使用时序数据
        self.use_history = use_history
        self.obs_his = U.obs_history(his_lenth=windows_size)
        
        self._reward_mech = 'global'
        self.timestep = 0
        self.hist = None
        self.ax = None
        self.obs_comm_matrix = None
        self.target_list = []
        self.target_dis = np.zeros((self.n_agents, 1))
        self.cos_sim = np.zeros((self.n_agents, 1))
        self.epislon = 1e-7
        self.collide_dis = 3.0
        self.bear_dis = 1.5
        self.safe_dis = 9.0
        if self.obs_mode == 'sum_obs_learn_comm':
            self.world.dim_c = 1
        # self.seed()

    @property
    def share_observation_space(self):
        share_obs_space = {}
        shape = self.agents[0].observation_space.shape
        for agent_id in range(self.nr_agents):
            share_obs_space[agent_id] = spaces.Box(low=-np.float32(np.inf), high=np.float32(np.inf), 
                                                   shape=(shape[0]*self.nr_agents, ), dtype=np.float32)
        return share_obs_space

    @property
    def observation_space(self):
        obs_space = {}
        for agent_id in range(self.nr_agents):
            obs_space[agent_id] = self.agents[agent_id].observation_space
        return obs_space

    def get_state(self,obs):
        share_obs = np.array(obs).reshape(1, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.nr_agents, axis=1)
        share_obs=share_obs.reshape(self.nr_agents,-1)
        return share_obs.copy()

    @property
    def state_space(self):
        return spaces.Box(low=-10., high=10., shape=(self.nr_agents * 3,), dtype=np.float32)

    @property
    def action_space(self):
        act_space = {}
        for agent_id in range(self.nr_agents):
            act_space[agent_id] = self.agents[agent_id].action_space
        return act_space

    @property
    def reward_mech(self):
        return self.reward_mech

    @property
    def agents(self):
        return self.world.policy_agents

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        np.random.seed(seed)

    @property
    def timestep_limit(self):
        return 512

    @property
    def is_terminal(self):
        if self.timestep >= self.timestep_limit:
            if self.ax:
                plt.close()
            return True
        return False

    def reset(self):
        if self.use_history:
            self.obs_his.clear_obs()
        self.timestep = 0
        # self.ax = None
        
        # 记录到达目标点的集合
            
        self.world.agents = [
            NavigateAgent(self)
            for _ in
            range(self.nr_agents)
        ]

        self.obs_comm_matrix = self.obs_radius * np.ones([self.nr_agents + 1, self.nr_agents + 1])
        self.obs_comm_matrix[0:-self.int_points_num, 0:-self.int_points_num] = self.comm_radius

        navigators = np.random.rand(self.nr_agents, 3)
        navigators[:, 0:2] = self.world_size * ((0.95 - 0.05) * navigators[:, 0:2] + 0.05)
        navigators[:, 2:3] = 2 * np.pi * navigators[:, 2:3]

        # int_points = (0.95 - 0.05) * np.random.rand(self.int_points_num, 2) + 0.05
        # int_points = self.world_size * int_points
        int_points = []
        while len(int_points) < self.int_points_num:
            new_point = (0.95 - 0.05) * np.random.rand(2) + 0.05
            new_point *= self.world_size
            too_close = any(np.linalg.norm(new_point - p) < self.safe_dis for p in int_points)
            if not too_close:
                int_points.append(new_point)
        int_points = np.array(int_points)
                
        self.world.agent_states = navigators
        self.world.landmark_states = int_points
        self.world.reset()
        self.distribute_target()
        
        if self.obs_radius < self.world_size * np.sqrt(2):
            sets = self.graph_feature()

        feats = [p.graph_feature for p in self.agents]

        if self.world.dim_c > 0:
            messages = np.zeros([self.nr_agents, 1])
        else:
            messages = []

        obs = []

        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     np.zeros([self.nr_agents, 2])
                                     )
            # ob[1] = self.timestep
            # ob[0] = i
            obs.append(ob)
        s_obs = np.array(obs)
        s_obs = np.reshape(s_obs, (self.agents[0].observation_space.shape[0] * self.nr_agents, ))
        s_obs = [s_obs for _ in range(self.nr_agents)]
        if self.use_history:
            self.obs_his.insert_obs((obs, s_obs))
            (obs, s_obs) = self.obs_his.get_obs()
        return obs, s_obs, None

    def step(self, actions):

        self.timestep += 1
        assert len(actions) == self.nr_agents
        clipped_actions = np.clip(actions, self.agents[0].action_space.low, self.agents[0].action_space.high)

        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action[0:2]
            if self.world.dim_c > 0:
                agent.action.c = action[2:]
        self.world.step()

        if self.obs_radius < self.world_size * np.sqrt(2):
            sets = self.graph_feature()

        feats = [p.graph_feature for p in self.agents]

        if self.world.dim_c > 0:
            messages = clipped_actions[:, 2:]
        else:
            messages = []

        velocities = np.vstack([agent.state.w_vel for agent in self.agents])

        next_obs = []

        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     velocities
                                     )
            # ob[1] = self.timestep
            # ob[0] = i
            next_obs.append(ob)
        rewards = self.get_reward_new(actions)
        # rewards = self.get_reward(actions)

        done = self.is_terminal
        '''
        if rewards[0] > -1 / self.obs_radius:  # distance of 1 in world coordinates, scaled by the reward scaling factor
            done = True
        '''
        # if done and self.timestep < self.timestep_limit:
        #     rewards = 100 * np.ones((self.nr_agents,))
        # info = dict()
        info = {'pursuer_states': self.world.agent_states,
                'evader_states': self.world.landmark_states,
                'state': np.vstack([self.world.agent_states[:, 0:2], self.world.landmark_states]),
                'actions': actions}

        # state=self.get_state(next_obs)
        s_obs = np.array(next_obs)
        s_obs = np.reshape(s_obs, (self.agents[0].observation_space.shape[0] * self.nr_agents, ))
        s_obs = [s_obs for _ in range(self.nr_agents)]
        if self.use_history:
            self.obs_his.insert_obs((next_obs, s_obs))
            (next_obs, s_obs) = self.obs_his.get_obs()
        return next_obs, s_obs, rewards, done, info, None
    
    def get_reward(self, actions):  
        dis = self.world.distance_matrix[0:-self.int_points_num]
        dis = dis[:, -self.int_points_num:]
        agent_dis = self.world.distance_matrix[0:-self.int_points_num]
        agent_dis = agent_dis[:, :-self.int_points_num]

        angle = self.world.angle_matrix[:, -self.int_points_num:]
        r = np.zeros((self.nr_agents,1))
        return_r = np.ones((self.nr_agents,1))
        # energy_rate = 0.1
        move_rew_rate = 1.0
        move_angle_rate = 0.002

        collide_rew_rate1 = -0.6#-1.0
        collide_rew_rate2 = -0.9#-2.0
        
        get_target_rew = 1.5# 10.0
        vec_norm_list = []
        for i in range(self.n_agents):
            # + 表示更近
            move_dis = self.target_dis[i] - dis[i][self.target_list[i]]
            r[i] += move_dis * move_rew_rate
            self.target_dis[i] = dis[i][self.target_list[i]]
            
            vec1 = self.agents[i].state.p_vel
            vec_norm_list.append(np.linalg.norm(vec1))
            pole = angle[i][self.target_list[i]]
            vec2 = [np.cos(pole), np.sin(pole)]
            cos_sim = vec1.dot(vec2) / (vec_norm_list[i] * np.linalg.norm(vec2) + self.epislon)
            angle_dis = self.cos_sim[i] - cos_sim
            r[i] += angle_dis * move_angle_rate
            self.cos_sim[i] = cos_sim
            
            # r[i] -= self.agents[i].energy_consumption * energy_rate
        reach_target_set = set()
        for j in range(self.n_agents):
            for i in range(self.int_points_num):
                if dis[j][i] <= self.bear_dis:
                    if i == self.target_list[j]:
                        reach_target_set.add(j)
                        r[j] += get_target_rew
                        break
        # 避障
        for j in range(self.n_agents):
            for i in range(self.n_agents):
                if agent_dis[j][i] < self.collide_dis and j != i \
                    and (j not in reach_target_set) and (i not in reach_target_set):
                    if agent_dis[j][i] > 1.0 :
                        collide_dis = - agent_dis[j][i] ** 2 + 2 * agent_dis[j][i] + 3.0
                        r[j] += collide_dis * collide_rew_rate1
                    else:
                        collide_dis = 1/(agent_dis[j][i] + 0.5) + 3.0
                        r[j] += collide_dis * collide_rew_rate2
        reward_rate = np.mean(r)
        return_r *= reward_rate
        return return_r
    
    def get_reward_old(self, actions):
        weight = np.array([5, 5])
        covered_flag = np.array([0, 0])
        now_covered = np.array([0, 0])
        cover_num = 0
        multi_rew = 10.0
        get_target_rew = 5.0
        bear_dis = 1.0
        energy_rate = 1.0
        dis = self.world.distance_matrix[0:-self.int_points_num]
        dis = dis[:, -self.int_points_num:]
        r = np.zeros((self.nr_agents,1))
        for j in range(self.n_agents):
            for i in range(self.int_points_num):
                if dis[j][i] <= bear_dis and now_covered[i] < weight[i]:
                    cover_num += 1
                    now_covered[i] += 1
                    covered_flag[i] = 1
                    r[j] += get_target_rew
                    break
        r += np.sum(covered_flag) * multi_rew
        for i, a in enumerate(self.agents):
            r[i] -= a.energy_consumption * energy_rate      
        return r
    
    def get_reward_new(self, actions):  
        dis = self.world.distance_matrix[0:-self.int_points_num]
        dis = dis[:, -self.int_points_num:]
        agent_dis = self.world.distance_matrix[0:-self.int_points_num]
        agent_dis = agent_dis[:, :-self.int_points_num]

        angle = self.world.angle_matrix[:, -self.int_points_num:]
        r = np.zeros((self.nr_agents,1))
        return_r = np.ones((self.nr_agents,1))
        move_rew_rate = 1.0
        move_angle_rate = 0.002

        collide_rew_rate1 = -2.0
        collide_rew_rate2 = -3.0
        
        get_target_rew = 5.0
        vec_norm_list = []
        for i in range(self.n_agents):
            # + 表示更近
            move_dis = self.target_dis[i] - dis[i][self.target_list[i]]
            r[i] += move_dis * move_rew_rate
            self.target_dis[i] = dis[i][self.target_list[i]]
            
            vec1 = self.agents[i].state.p_vel
            vec_norm_list.append(np.linalg.norm(vec1))
            pole = angle[i][self.target_list[i]]
            vec2 = [np.cos(pole), np.sin(pole)]
            cos_sim = vec1.dot(vec2) / (vec_norm_list[i] * np.linalg.norm(vec2) + self.epislon)
            angle_dis = self.cos_sim[i] - cos_sim
            r[i] += angle_dis * move_angle_rate
            self.cos_sim[i] = cos_sim
            
        reach_target_set = set()
        for j in range(self.n_agents):
            for i in range(self.int_points_num):
                if dis[j][i] <= self.bear_dis:
                    if i == self.target_list[j]:
                        reach_target_set.add(j)
                        r[j] += get_target_rew
                        break
        # 避障
        for j in range(self.n_agents):
            for i in range(self.n_agents):
                if agent_dis[j][i] < self.collide_dis and j != i \
                    and (j not in reach_target_set) and (i not in reach_target_set):
                    if agent_dis[j][i] > 1.0 :
                        collide_dis = - agent_dis[j][i] ** 2 + 2 * agent_dis[j][i] + 3.0
                        r[j] += collide_dis * collide_rew_rate1
                    else:
                        collide_dis = 1/(agent_dis[j][i] + 0.5) + 3.0
                        r[j] += collide_dis * collide_rew_rate2
        reward_rate = np.mean(r)
        return_r *= reward_rate
        return return_r

    def graph_feature(self):
        adj_matrix = np.array(self.world.distance_matrix < self.obs_comm_matrix, dtype=float)
        # visibles = np.sum(adj_matrix, axis=0) - 1
        # print("mean neighbors seen: ", np.mean(visibles[:-1]))
        # print("evader seen by: ", visibles[-1])
        sets = U.dfs(adj_matrix, 2)

        g = nwx.Graph()

        for set_ in sets:
            l_ = list(set_)
            if self.nr_agents in set_:
                # points = self.nodes[set_, 0:2]
                # dist_matrix = self.get_euclid_distances(points, matrix=True)

                # determine distance and adjacency matrix of subset
                dist_matrix = np.array([self.world.distance_matrix[x] for x in list(itertools.product(l_, l_))]).reshape(
                        [len(l_), len(l_)])

                obs_comm_matrix = np.array(
                    [self.obs_comm_matrix[x] for x in list(itertools.product(l_, l_))]).reshape(
                    [len(l_), len(l_)])

                adj_matrix_sub = np.array((0 <= dist_matrix) & (dist_matrix < obs_comm_matrix), dtype=float)
                connection = np.where(adj_matrix_sub == 1)
                edges = [[x[0], x[1]] for x in zip([l_[c] for c in connection[0]], [l_[c] for c in connection[1]])]

                g.add_nodes_from(l_)
                g.add_edges_from(edges)
                for ind, e in enumerate(edges):
                    g[e[0]][e[1]]['weight'] = dist_matrix[connection[0][ind], connection[1][ind]]

        for i in range(self.nr_agents):
            try:
                self.agents[i].graph_feature = \
                    nwx.shortest_path_length(g, source=i, target=self.nr_agents, weight='weight')
            except:
                self.agents[i].graph_feature = np.inf

        return sets

    def render(self, mode='human'):
        if mode == 'animate':
            output_dir = "/tmp/video/"
            if self.timestep == 0:
                import shutil
                import os
                try:
                    shutil.rmtree(output_dir)
                except FileNotFoundError:
                    pass
                os.makedirs(output_dir, exist_ok=True)

        if not self.ax:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlim((0, self.world_size))
            ax.set_ylim((0, self.world_size))
            self.ax = ax

        else:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.set_xlim((0, self.world_size))
            self.ax.set_ylim((0, self.world_size))

        comm_circles = []
        obs_circles = []
        self.ax.scatter(self.world.landmark_states[:, 0], self.world.landmark_states[:, 1], c='r', s=20)
        self.ax.scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=20)
        for i in range(self.nr_agents):
            comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                       self.world.agent_states[i, 1]),
                                      self.comm_radius, color='g', fill=False))
            self.ax.add_artist(comm_circles[i])

            obs_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.obs_radius, color='g', fill=False))
            self.ax.add_artist(obs_circles[i])

        if mode == 'human':
            plt.pause(0.01)
        elif mode == 'animate':
            if self.timestep % 1 == 0:
                plt.savefig(output_dir + format(self.timestep//1, '04d'))

            if self.is_terminal:
                import os
                os.system("ffmpeg -r 10 -i " + output_dir + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4")

    def distribute_target(self):
        self.target_list = []
        self.target_dis = np.zeros((self.n_agents, 1))
        self.cos_sim = np.zeros((self.n_agents, 1))
        aver_num = self.n_agents//self.int_points_num
        remain = self.n_agents % self.int_points_num
        for i in range(self.int_points_num):
            for j in range(aver_num):
                self.target_list.append(i)
        for i in range(remain):
            self.target_list.append(self.int_points_num-1)    
        dis = self.world.distance_matrix[0:-self.int_points_num]
        dis = dis[:, -self.int_points_num:]

        angle = self.world.angle_matrix[:, -self.int_points_num:]
        for i in range(self.n_agents):
            self.agents[i].target = self.target_list[i]
            self.target_dis[i] = dis[i][self.target_list[i]]
            vec1 = self.agents[i].state.p_vel
            pole = angle[i][self.target_list[i]]
            vec2 = [np.cos(pole), np.sin(pole)]
            cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + self.epislon)
            self.cos_sim[i] = cos_sim