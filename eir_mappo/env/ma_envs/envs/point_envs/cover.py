import numpy as np
import gym
import random
from gym import spaces
from gym.utils import seeding
from eir_mappo.env.ma_envs.commons.utils import EzPickle
from eir_mappo.env.ma_envs import base
# from ma_envs.envs.environment import MultiAgentEnv
from eir_mappo.env.ma_envs.agents.point_agents.cover_agent import CoverAgent
from eir_mappo.env.ma_envs.commons import utils as U
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

def generate_pois(n_major=3, n_poi_per_major=10, map_size=100, min_distance_between_majors=20,min_distance_between_pois=5):
    def is_far_enough(new_point, existing_points, min_distance):
        return all(np.linalg.norm(np.array(new_point) - np.array(point)) >= min_distance for point in existing_points)

    # 生成主要点
    major_points = []
    while len(major_points) < n_major:
        point = [random.uniform(0, map_size), random.uniform(0, map_size)]
        if not major_points or is_far_enough(point, major_points, min_distance_between_majors):
            major_points.append(point)

    # 为每个主要点生成周围的兴趣点
    int_points = []
    for major_point in major_points:
        for _ in range(n_poi_per_major):
            offset = np.random.normal(0, 5, 2) 
            # 下面是原先30个点的时候 用的 但是现在只有15个点 所以改成了3
            # offset = np.random.normal(0, 5, 2)  # 假设标准差为5
            poi = np.array(major_point) + offset
            # 确保兴趣点在地图范围内
            poi = np.clip(poi, 0, map_size)
            int_points.append(poi)
            if not int_points or is_far_enough(poi, int_points, min_distance_between_pois):
                int_points.append(poi)
                break

    return np.array(major_points), np.array(int_points)

'''
强制把智能体位置放在一个区域的生成兴趣点的方式：
'''

# def generate_pois(n_major=3, n_poi_per_major=10, map_size=100, min_distance_between_majors=20, gaussian_std_dev=6):
#     def clip(value, min_val, max_val):
#         return max(min_val, min(value, max_val))

#     # 生成主要点
#     major_points = []
#     while len(major_points) < n_major:
#         point = [random.uniform(0.3 * map_size, 0.7 * map_size), random.uniform(0.3 * map_size, 0.7 * map_size)]
#         if not major_points or all(np.linalg.norm(np.array(point) - np.array(existing_point)) >= min_distance_between_majors for existing_point in major_points):
#             major_points.append(point)

#     # 为每个主要点生成周围的兴趣点
#     int_points = []
#     for major_point in major_points:
#         for _ in range(n_poi_per_major):
#             x_diff = np.random.normal(0, gaussian_std_dev)
#             y_diff = np.random.normal(0, gaussian_std_dev)
#             x_poi = clip(major_point[0] + x_diff, 0, map_size)
#             y_poi = clip(major_point[1] + y_diff, 0, map_size)
#             int_points.append([x_poi, y_poi])

#     return np.array(major_points), np.array(int_points)

class CoverEnv_ori(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self,
                 windows_size=1,
                 use_history=False,
                 nr_pursuers=5,
                 int_points_num=30,
                 obs_mode='2D_rbf',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=8,
                 bearing_bins=8,
                 torus=True,
                 dynamics='direct'):
        # torus 为 true 时，表示环境是一个环形环境，智能体可以从环的一侧穿过环的另一侧。
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
            CoverAgent(self) for _ in
            range(self.nr_agents)
        ]
        self.world.int_points = []
        # 使用时序数据
        self.use_history = use_history
        self.obs_his = U.obs_history(his_lenth=windows_size)
        
        self._reward_mech = 'global'
        self.timestep = None
        self.hist = None
        self.ax = None
        self.obs_comm_matrix = None
        self.target_list = []
        self.target_dis = np.zeros((self.n_agents, 1))
        self.cos_sim = np.zeros((self.n_agents, 1))
        self.epislon = 1e-7
        self.collide_dis = 3.0
        # tmc的地图是200*200的，cover_range是10，这里用的是5
        self.cover_range = 5
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
        self.timestep = 0
        if self.use_history:
            self.obs_his.clear_obs()
        # self.ax = None
        self.world.agents = [
            CoverAgent(self)
            for _ in
            range(self.nr_agents)
        ]

        # 定义智能体之间的观测和通信范围
        self.obs_comm_matrix = self.obs_radius * np.ones([self.nr_agents + 1, self.nr_agents + 1])
        self.obs_comm_matrix[0:-self.int_points_num, 0:-self.int_points_num] = self.comm_radius
        
        # 旧版本初始化智能体的位置和方向角
        navigators = np.random.rand(self.nr_agents, 3)
        # 避免智能体初始位置过于靠边 
        navigators[:, 0:2] = self.world_size * ((0.95 - 0.05) * navigators[:, 0:2] + 0.05)
        navigators[:, 2:3] = 2 * np.pi * navigators[:, 2:3]

        # 初始化生成兴趣点
        self.major_points, self.int_points = generate_pois()
        self.world.agent_states = navigators
        self.world.landmark_states = self.int_points
        self.world.reset()
        # self.distribute_target()
        
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
            next_obs.append(ob)

        rewards = self.get_reward(actions)

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
        dis_to_int_points = self.world.distance_matrix[0:-self.int_points_num]
        dis_to_int_points = dis_to_int_points[:, -self.int_points_num:]
        
        r = np.zeros((self.nr_agents, 1))

        cover_reward = 15.0  
        # repeat_cover_penalty = -5.0  
        energy_penalty_rate = -0.05
        # cooperation_reward = 1.0  
        gather_penalty = -10.0  
        # cooperation_radius=20
        gather_radius=8

        # 计算已覆盖的兴趣点数量
        covered_points = set()
        for i in range(self.nr_agents):
            for j in range(self.int_points_num):
                if dis_to_int_points[i][j] < self.cover_range:
                    covered_points.add(j)

        # 计算覆盖比率
        cover_ratio = len(covered_points) / self.int_points_num

        for i in range(self.nr_agents):
            r[i] += cover_reward * cover_ratio / self.nr_agents

            # 能源消耗惩罚
            action = actions[i]  # 假设 actions 包含了每个智能体的行动信息
            velocity = self.agents[i].state.p_vel  # 假设每个智能体有速度属性
            energy_consumed = self.calculate_energy(action, velocity)
            r[i] += energy_penalty_rate * energy_consumed

        # 避免聚集惩罚
        if self.is_agents_gathered(self.agents,gather_radius):
            for i in range(self.nr_agents):
                r[i] += gather_penalty

        return r

    # 判断是否有过多的智能体聚集在一起
    def is_agents_gathered(self, agents, gather_radius, threshold=2):
        for agent in agents:
            nearby_agents = sum(np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos) < gather_radius
                                for other_agent in agents if other_agent != agent)
            if nearby_agents > threshold:
                return True
        return False

    def calculate_energy(self, action, velocity):
        # 假设能源消耗与速度和行动大小成比例
        # action 代表智能体的移动方向和幅度
        # velocity 代表智能体的当前速度

        distance_factor = np.linalg.norm(action)  # 行动引起的移动距离
        speed_factor = np.linalg.norm(velocity)  # 当前速度

        # 假定基础能源消耗和额外因速度增加的能源消耗
        base_energy_rate = 0.05
        speed_energy_rate = 0.1

        # 计算能源消耗
        energy_consumed = base_energy_rate * distance_factor + speed_energy_rate * speed_factor
        return energy_consumed





    # # 目前egnn能训练出来的奖励函数版本：30个兴趣点
    # def get_reward(self, actions):
    #     dis_to_int_points = self.world.distance_matrix[0:-self.int_points_num]
    #     dis_to_int_points = dis_to_int_points[:, -self.int_points_num:]
        
    #     r = np.zeros((self.nr_agents, 1))

    #     # 整体覆盖兴趣点的奖励 
    #     cover_reward = 15.0  
    #     # repeat_cover_penalty = -5.0  
    #     energy_penalty_rate = -0.05
    #     # cooperation_reward = 1.0  
    #     gather_penalty = -10.0  
    #     # cooperation_radius=20
    #     gather_radius=8

    #     # 计算已覆盖的兴趣点数量
    #     covered_points = set()  
    #     for i in range(self.nr_agents):
    #         for j in range(self.int_points_num):
    #             if dis_to_int_points[i][j] < self.cover_range:
    #                 covered_points.add(j)

    #     # 计算覆盖比率
    #     cover_ratio = len(covered_points) / self.int_points_num

    #     for i in range(self.nr_agents):
    #         r[i] += cover_reward * cover_ratio / self.nr_agents

    #         # 在base.py当中查看相关动力学设置 velocity 是 action的扩大了十倍的结果 
           
    #         # 能源消耗惩罚
    #         action = actions[i]  
    #         velocity = self.agents[i].state.p_vel 
    #         energy_consumed = self.calculate_energy(action, velocity)
    #         r[i] += energy_penalty_rate * energy_consumed

    #     # 避免聚集惩罚
    #     if self.is_agents_gathered(self.agents,gather_radius):
    #         for i in range(self.nr_agents):
    #             r[i] += gather_penalty
                
    #     # for i in range(self.nr_agents):
    #     #     if i != 0:
    #     #         r[i] = 0

    #     return r

    # # 判断是否有过多的智能体聚集在一起
    # def is_agents_gathered(self, agents, gather_radius, threshold=2):
    #     for agent in agents:
    #         nearby_agents = sum(np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos) < gather_radius
    #                             for other_agent in agents if other_agent != agent)
    #         if nearby_agents > threshold:
    #             return True
    #     return False

    # def calculate_energy(self, action, velocity):
    #     # 假设能源消耗与速度和行动大小成比例
    #     # action 代表智能体的移动方向和幅度
    #     # velocity 代表智能体的当前速度

    #     distance_factor = np.linalg.norm(action)  # 用第二范数计算行动引起的移动距离
    #     speed_factor = np.linalg.norm(velocity)  # 用第二范数当前速度

    #     # 超参数 
    #     base_energy_rate = 0.05
    #     speed_energy_rate = 0.1

    #     # 计算能源消耗
    #     energy_consumed = base_energy_rate * distance_factor + speed_energy_rate * speed_factor
    #     return energy_consumed
    


    # 目前的最新版本：综合考虑过的各种奖励和惩罚制度 
    '''
    def get_reward(self, actions):
        dis_to_int_points = self.world.distance_matrix[0:-self.int_points_num]
        dis_to_int_points = dis_to_int_points[:, -self.int_points_num:]
        
        r = np.zeros((self.nr_agents, 1))

        cover_reward = 15.0  
        repeat_cover_penalty = -10.0  
        energy_penalty_rate = -0.04
        gather_penalty = -40.0  
        gather_radius=8

        # 记录每个兴趣点被哪些智能体覆盖
        point_coverage = {j: set() for j in range(self.int_points_num)}

        # 计算已覆盖的兴趣点数量
        covered_points = set()
        for i in range(self.nr_agents):
            for j in range(self.int_points_num):
                if dis_to_int_points[i][j] < self.cover_range:
                    if i not in point_coverage[j]:
                        point_coverage[j].add(i)
                        covered_points.add(j)
                        r[i] += cover_reward
                    else:
                        r[i] += repeat_cover_penalty

        # 计算覆盖比率
        cover_ratio = len(covered_points) / self.int_points_num

        for i in range(self.nr_agents):
            r[i] += cover_reward * cover_ratio / self.nr_agents

            # 能源消耗惩罚
            action = actions[i]  # 假设 actions 包含了每个智能体的行动信息
            velocity = self.agents[i].state.p_vel  # 假设每个智能体有速度属性
            energy_consumed = self.calculate_energy(action, velocity)
            r[i] += energy_penalty_rate * energy_consumed

            # 合作奖励
            # for j in range(self.nr_agents):
            #     if i != j and self.is_in_cooperation_range(cooperation_radius,self.agents[i], self.agents[j]):
            #         r[i] += cooperation_reward

        # 避免聚集惩罚
        if self.is_agents_gathered(self.agents,gather_radius):
            for i in range(self.nr_agents):
                r[i] += gather_penalty / 2

        return r

    # 判断是否有过多的智能体聚集在一起
    def is_agents_gathered(self, agents, gather_radius, threshold=2):
        for agent in agents:
            nearby_agents = sum(np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos) < gather_radius
                                for other_agent in agents if other_agent != agent)
            if nearby_agents > threshold:
                return True
        return False
    
    #  判断两个智能体是否处于合作范围内
    # def is_in_cooperation_range(self,cooperation_radius,agent1, agent2 ):
    #     distance = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
    #     return distance < cooperation_radius

    def calculate_energy(self, action, velocity):
        # 假设能源消耗与速度和行动大小成比例
        # action 代表智能体的移动方向和幅度
        # velocity 代表智能体的当前速度

        distance_factor = np.linalg.norm(action)  # 行动引起的移动距离
        speed_factor = np.linalg.norm(velocity)  # 当前速度

        # 假定基础能源消耗和额外因速度增加的能源消耗
        base_energy_rate = 0.05
        speed_energy_rate = 0.1

        # 计算能源消耗
        energy_consumed = base_energy_rate * distance_factor + speed_energy_rate * speed_factor
        return energy_consumed
    '''
        

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
        angle = self.world.angle_matrix[0:-self.int_points_num]
        angle = dis[:, -self.int_points_num:]
        for i in range(self.n_agents):
            self.agents[i].target = self.target_list[i]
            self.target_dis[i] = dis[i][self.target_list[i]]
            vec1 = self.agents[i].state.p_vel
            pole = angle[i][self.target_list[i]]
            vec2 = [np.cos(pole), np.sin(pole)]
            cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + self.epislon)
            self.cos_sim[i] = cos_sim