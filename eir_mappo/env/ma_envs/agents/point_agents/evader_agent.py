import scipy.spatial as ssp
import numpy as np
import eir_mappo.env.ma_envs.commons.utils as U
import shapely.geometry as sg
from eir_mappo.env.ma_envs.base import Agent


class Evader(Agent):
    def __init__(self, experiment):
        super(Evader, self).__init__()
        self.obs_radius = experiment.obs_radius
        self.world_size = experiment.world_size
        self.torus = experiment.torus

        self.policy = experiment.evader_policy


        if self.policy != 'circle':
            self.dynamics = 'direct'
        else:
            self.dynamics = 'unicycle'
            self.max_lin_velocity = 8
            self.max_ang_velocity = self.max_ang_velocity / 3 * 2

        self.max_speed = 15  # cm/s
        if self.torus:
            self.bounding_box = np.array([0., 2 * self.world_size, 0., 2 * self.world_size])
        else:
            self.bounding_box = np.array([0., self.world_size, 0., self.world_size])

        self.action_callback = self.step

    def step(self, agent, world):
        if self.policy == 'tac':
            return self.step_tac(agent, world)
        if self.policy == 'nav':
            return np.array([0.0,0.0])
        if self.policy == 'square':
            return self.step_square()
        if self.policy == 'circle':
            return self.step_circle()
        if self.policy == 'apf':
            return self.step_apf(agent, world)

    def reset_tac(self, state):
        self.state.p_pos = state
        self.state.p_vel = np.zeros(2)

    def reset_apf(self, state):
        self.state.p_pos = state
        self.state.p_vel = np.zeros(2)


    def reset_square(self):

        self.state.p_pos = np.array([0.25 * self.world_size, 0.75 * self.world_size])


    def reset_circle(self):
        self.state.p_pos = np.array([0.5 * self.world_size, 0.9 * self.world_size])
        self.state.p_vel = np.zeros(2)
        self.state.p_orientation = 0

    def step_apf(self,agent,world):
        pursuer_cor=world.agent_states[:, 0:2]
        evader_cor=self.state.p_pos
        dis=U.get_distances(evader_cor,pursuer_cor)

        dif=pursuer_cor-evader_cor
        v=[]
        for i in range(world.nr_agents):
            v.append(dif[i]/(dis[i]*dis[i]))
        v=np.array(v)

        v=np.sum(v,axis=0)
        print(v)
        return v



    def step_square(self):

        act = 0.5
        limit = 0.1 * act * self.max_speed
        x = self.state.p_pos[0]
        y = self.state.p_pos[1]
        if (y == 0.75 * self.world_size) & (x >= 0.25 * self.world_size):
            action = [act, 0]

        if (x == 0.75 * self.world_size) & (y <= 0.75 * self.world_size):
            action = [0, -act]

        if (x <= 0.75 * self.world_size) & (y == 0.25 * self.world_size):
            action = [-act, 0]

        if (x == 0.25 * self.world_size) & (y < 0.75 * self.world_size):
            action = [0, act]

        return np.array(action)

    def step_circle(self):
        angular = self.max_speed / self.world_size * 0.25
        return angular


    def step_tac(self, agent, world):
        if self.torus:
            points_center = np.vstack([world.agent_states[:, 0:2], self.state.p_pos])
            pursuers_down_right = np.hstack([world.agent_states[:, 0:1] + world.world_size, world.agent_states[:, 1:2]])
            pursuers_up_left = np.hstack([world.agent_states[:, 0:1], world.agent_states[:, 1:2] + world.world_size])
            pursuers_up_right = np.hstack(
                [world.agent_states[:, 0:1] + world.world_size, world.agent_states[:, 1:2] + world.world_size])
            evader_down_right = np.hstack([self.state.p_pos[0:1] + world.world_size, self.state.p_pos[1:2]])
            evader_up_left = np.hstack([self.state.p_pos[0:1], self.state.p_pos[1:2] + world.world_size])
            evader_up_right = np.hstack(
                [self.state.p_pos[0:1] + world.world_size, self.state.p_pos[1:2] + world.world_size])
            points_down_right = np.hstack([points_center[:, 0:1] + world.world_size, points_center[:, 1:2]])
            points_up_left = np.hstack([points_center[:, 0:1], points_center[:, 1:2] + world.world_size])
            points_up_right = np.hstack(
                [points_center[:, 0:1] + world.world_size, points_center[:, 1:2] + world.world_size])

            nodes = np.vstack([world.agent_states[:, 0:2],
                               pursuers_down_right,
                               pursuers_up_left,
                               pursuers_up_right,
                               self.state.p_pos,
                               evader_down_right,
                               evader_up_left,
                               evader_up_right])

            dist_matrix_full = U.get_euclid_distances(nodes)

            quadrant_check = np.sign(self.state.p_pos - world.world_size / 2)
            if np.all(quadrant_check == np.array([1, 1])):
                evader_quadrant = 0
            elif np.all(quadrant_check == np.array([-1, 1])):
                evader_quadrant = 1
            elif np.all(quadrant_check == np.array([1, -1])):
                evader_quadrant = 2
            elif np.all(quadrant_check == np.array([-1, -1])):
                evader_quadrant = 3

            evader_dist = dist_matrix_full[:-4, -4 + evader_quadrant]
            sub_list = list(np.where(evader_dist < self.obs_radius)[0])
            if len(sub_list) > 10:
                sub_list = list(np.argsort(evader_dist)[0:10])
            sub_list.append(4 * world.nr_agents + evader_quadrant)
            evader_sub = len(sub_list) - 1
            closest_pursuer = np.where(evader_dist == evader_dist.min())[0]

            nodes_center_sub = nodes[sub_list, :]
            nodes_left = np.copy(nodes_center_sub)
            nodes_left[:, 0] = self.bounding_box[0] - (nodes_left[:, 0] - self.bounding_box[0])
            nodes_right = np.copy(nodes_center_sub)
            nodes_right[:, 0] = self.bounding_box[1] + (self.bounding_box[1] - nodes_right[:, 0])
            nodes_down = np.copy(nodes_center_sub)
            nodes_down[:, 1] = self.bounding_box[2] - (nodes_down[:, 1] - self.bounding_box[2])
            nodes_up = np.copy(nodes_center_sub)
            nodes_up[:, 1] = self.bounding_box[3] + (self.bounding_box[3] - nodes_up[:, 1])

            points = np.vstack([nodes_center_sub, nodes_down, nodes_left, nodes_right, nodes_up])

        else:
            nodes = np.vstack([world.agent_states[:, 0:2],
                               self.state.p_pos,
                               ])
            distances = U.get_euclid_distances(nodes)
            evader_dist = distances[-1, :-1]
            closest_pursuer = np.where(evader_dist == evader_dist.min())[0]
            sub_list = list(np.where(evader_dist < self.obs_radius)[0])
            if len(sub_list) > 10:
                sub_list = list(np.argsort(evader_dist)[0:10])
            sub_list.append(world.nr_agents)
            evader_sub = len(sub_list) - 1

            nodes_center_sub = nodes[sub_list, :]
            nodes_left = np.copy(nodes_center_sub)
            nodes_left[:, 0] = self.bounding_box[0] - (nodes_left[:, 0] - self.bounding_box[0])
            nodes_right = np.copy(nodes_center_sub)
            nodes_right[:, 0] = self.bounding_box[1] + (self.bounding_box[1] - nodes_right[:, 0])
            nodes_down = np.copy(nodes_center_sub)
            nodes_down[:, 1] = self.bounding_box[2] - (nodes_down[:, 1] - self.bounding_box[2])
            nodes_up = np.copy(nodes_center_sub)
            nodes_up[:, 1] = self.bounding_box[3] + (self.bounding_box[3] - nodes_up[:, 1])

            points = np.vstack([nodes_center_sub, nodes_down, nodes_left, nodes_right, nodes_up])

        vor = ssp.Voronoi(points)

        d = np.zeros(2)

        for i, ridge in enumerate(vor.ridge_points):
            if evader_sub in set(ridge) and np.all([r <= evader_sub for r in ridge]):
                if self.torus:
                    neighbor = min([sub_list[r] for r in ridge])
                else:
                    # neighbor = min(ridge)
                    neighbor = min([sub_list[r] for r in ridge])

                if neighbor in closest_pursuer:
                    ridge_inds = vor.ridge_vertices[i]
                    a = vor.vertices[ridge_inds[0], :]
                    b = vor.vertices[ridge_inds[1], :]

                    line_of_control = b - a
                    L_i = np.linalg.norm(line_of_control)

                    if self.torus:
                        xi = nodes[neighbor, :] - nodes[4 * world.nr_agents + evader_quadrant]
                    else:
                        xi = nodes[neighbor, :] - self.state.p_pos
                    eta_h_i = xi / np.linalg.norm(xi)
                    eta_v_i = np.array([-eta_h_i[1], eta_h_i[0]])

                    if self.torus:
                        line1 = sg.LineString([nodes[4 * world.nr_agents + evader_quadrant], nodes[neighbor, :]])
                    else:
                        line1 = sg.LineString([self.state.p_pos, nodes[neighbor, :]])
                    line2 = sg.LineString([a, b])
                    intersection = line1.intersection(line2)

                    if not intersection.is_empty:
                        inter_point = np.hstack(intersection.xy)

                        if np.dot(line_of_control, eta_v_i.flatten()) > 0:
                            l_i = np.linalg.norm(a - inter_point)
                        else:
                            l_i = np.linalg.norm(b - inter_point)
                    else:
                        if np.dot(line_of_control, eta_v_i.flatten()) > 0:
                            l_i = 0
                        else:
                            l_i = L_i

                    alpha_h_i = - L_i / 2
                    alpha_v_i = (l_i ** 2 - (L_i - l_i) ** 2) / (2 * np.linalg.norm(xi))

                    d = (alpha_h_i * eta_h_i - alpha_v_i * eta_v_i) / np.sqrt(alpha_h_i ** 2 + alpha_v_i ** 2)

        assert ('d' in locals())

        return d

    def reset(self, state):
        if self.policy == 'tac':
            self.reset_tac(state)
        if self.policy == 'nav':
            self.reset_tac(state)
        if self.policy == 'square':
            self.reset_square()
        if self.policy == 'circle':
            self.reset_circle()
        if self.policy == 'apf':
            self.reset_apf(state)