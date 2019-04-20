import numpy as np
import math
from gym.envs.classic_control import rendering

# Environment

## Perfect sensing (i.e., relative positions and radii of N nearest obstacles)
def _o_env_pose(self):
    get = _get_o_env_pose.__get__(self)

    # For self.observation_space
    one_obs_pos = np.ones((2, self.hp_dim))
    one_obs_pos[0, :] = -2 * self.hp_bound
    one_obs_pos[1, :] = 2 * self.hp_bound
    one_obs_r = np.ones((2, 1))
    one_obs_r[:, 0] = self.hp_obs_r
    one_obs = np.hstack((one_obs_pos, one_obs_r))
    lim = np.hstack([one_obs for _ in range(self.hp_n_nearest_obs)])

    vis = _vis_o_env_pose.__get__(self)
    return get, lim, vis

def _get_o_env_pose(self, i):
    p = self.s_uav_pos[i]
    # Unify boundary surfaces and normal obstacles by creating "proxy obstacles" with zero radii
    bound_pos = np.array([p for _ in range(2 * self.hp_dim)])  # Number of proxy obstacles is 2*dim
    for di in range(self.hp_dim):
        bound_pos[2 * di][di] = -self.hp_bound[di]
        bound_pos[2 * di + 1][di] = self.hp_bound[di]
    bound_r = np.zeros(bound_pos.shape[0])

    rela_pos = np.vstack([self.s_obs_pos - p, bound_pos - p])
    r = np.hstack([self.s_obs_r, bound_r])

    # Find n nearest obstacles
    dis = np.linalg.norm(rela_pos, axis=1) - r
    order_mask = np.argsort(dis)[:self.hp_n_nearest_obs]

    res_pos = np.zeros((self.hp_n_nearest_obs, self.hp_dim))  # Considering the case where number of obstacles < self.hp_n_nearest_obs, set default to 0, because it's impossible in reality
    res_r = np.zeros(self.hp_n_nearest_obs)  # Although 0 radius is possible (boundary), invalidity can be inferred via res_pos
    res_pos[:order_mask.shape[0]] = rela_pos[order_mask]
    res_r[:order_mask.shape[0]] = r[order_mask]

    # Represent the vectors in the local coordinate
    rela_goal_pos = self.s_goal - self.s_uav_pos[i]
    rot_mat = self.global_to_local(rela_goal_pos)
    trans_res_pos = np.dot(rot_mat, res_pos.T)

    if self.hp_coord_mode == 'local':
        return np.hstack([trans_res_pos.T, np.expand_dims(res_r, axis=1)])
    else:
        return np.hstack([res_pos, np.expand_dims(res_r, axis=1)])

def _vis_o_env_pose(self):
    if self.vis_init:
        self.vis_env = []
        env_color = (230./255., 162./255., 227./255.)
        for _ in range(self.hp_uav_n):
            lines = []
            for _ in range(self.hp_n_nearest_obs):
                line = rendering.Line()
                line.set_color(*env_color)
                self.vis_viewer.add_geom(line)
                lines.append(line)
            self.vis_env.append(lines)

    for i in range(self.hp_uav_n):
        rela_goal_pos = self.s_goal - self.s_uav_pos_prev[i]    # Considering coordinate transformation, uavs' positions at last time step are required
        rot_inv_mat = self.local_to_global(rela_goal_pos)

        # Compute the line segment from observation (i.e., [rela_pos, r] of obstacles)
        local_obs_rela_pos = self.o_env_prev[i][:, :-1]     # The last element is radius, which is a scalar independent of coordinate
        if self.hp_coord_mode == 'local':
            obs_rela_pos = np.dot(rot_inv_mat, local_obs_rela_pos.T).T
        else:
            obs_rela_pos = local_obs_rela_pos

        obs_r = self.o_env_prev[i][:, -1]
        norm = np.linalg.norm(obs_rela_pos, axis=1)
        dir_vec = obs_rela_pos / np.expand_dims(norm, axis=1)
        intersec_point_rela_pos = dir_vec * np.expand_dims(norm - obs_r, axis=1)
        intersec_point_pos = self.s_uav_pos_prev[i] + intersec_point_rela_pos
        for j in range(self.hp_n_nearest_obs):
            self.vis_env[i][j].start = self.s_uav_pos_prev[i]
            self.vis_env[i][j].end = intersec_point_pos[j]

## 2D lidar sensing
def _o_env_sensor(self):
    get = _get_o_env_sensor.__get__(self)

    sensor_dis = np.ones((2, 1))
    sensor_dis[0] = 0.
    sensor_dis[1] = self.hp_lidar_range
    sensor = np.hstack([sensor_dis for _ in range(self.hp_lidar_n)])

    angle = np.ones((2, 1))
    angle[0] = -1
    angle[1] = 1
    angle_func = np.hstack([angle for _ in range(2)])

    lim = np.hstack((sensor, angle_func))

    vis = _vis_o_env_sensor.__get__(self)
    return get, lim, vis

def _get_o_env_sensor(self, i):
    sensor_info = np.zeros((self.hp_lidar_n, self.hp_dim)) + self.hp_lidar_range # record the coordinate of sensor end
    s_uav_pos_i = self.s_uav_pos[i].copy()
    if self.hp_lidar_fixed_dir:     # The direction of lidar is always in [0., 1.]
        lidar_dir = np.array([0., 1.])
    else:   # The direction of lidar changes as the uav move and is the same as the moving direction of the uav
        lidar_dir = self.s_uav_v_prev[i].copy()

    # Initialize counter-clockwise point clouds starting at -\pi
    if self.hp_lidar_angle_min == -np.pi:
        sensor_theta = np.linspace(self.hp_lidar_angle_min, self.hp_lidar_angle_max, self.hp_lidar_n + 1)   # -\pi and \pi are logically identical
        sensor_theta = sensor_theta[:self.hp_lidar_n]
    else:
        sensor_theta = np.linspace(self.hp_lidar_angle_min, self.hp_lidar_angle_max, self.hp_lidar_n)
    x_sensor = (np.zeros((self.hp_lidar_n,)) + self.hp_lidar_range) * np.cos(sensor_theta)
    y_sensor = (np.zeros((self.hp_lidar_n,)) + self.hp_lidar_range) * np.sin(sensor_theta)
    xy_sensor = np.array([[x, y] for x, y in zip(x_sensor, y_sensor)])

    # Rotate point clouds such that the first point directed at the back
    if np.linalg.norm(lidar_dir) == 0:
        cos_rot = 1.
        sin_rot = 0.
    else:
        cos_rot = lidar_dir[0] / np.linalg.norm(lidar_dir)
        sin_rot = lidar_dir[1] / np.linalg.norm(lidar_dir)
    rotation = np.array([[cos_rot, sin_rot],[-sin_rot, cos_rot]])   # The real rotation matrix is rotation.T
    sensor_info = np.dot(xy_sensor, rotation) + s_uav_pos_i     # A matrix of points evenly distributed on a circle

    # Uav
    if self.hp_lidar_detect_uav:
        s_uav_pos = self.s_uav_pos
        res = np.delete(s_uav_pos, i, 0)
        for uav_partner in res:
            distance, intersections = matrix_segment_circle_intersection(self, s_uav_pos_i, sensor_info, uav_partner, self.hp_uav_r)
            sensor_info = intersections

    # Obstacle
    # Filt obstacles within sensing range first
    s_uav_pos_i_obs = np.zeros((len(self.s_obs_pos), self.hp_dim)) + s_uav_pos_i
    dis = np.linalg.norm(s_uav_pos_i_obs - self.s_obs_pos, axis=1)
    obs_ind = []
    for oi, disi in enumerate(dis):
        if disi < self.hp_lidar_range + self.s_obs_r[oi]:
            obs_ind.append(oi)

    for oi in obs_ind:
        distance, intersections = matrix_segment_circle_intersection(self, s_uav_pos_i, sensor_info, self.s_obs_pos[oi], self.s_obs_r[oi])
        sensor_info = intersections

    # Boundary
    win_coord = np.array([
        [-self.hp_bound[0], -self.hp_bound[1]],
        [self.hp_bound[0], -self.hp_bound[1]],
        [self.hp_bound[0], self.hp_bound[1]],
        [-self.hp_bound[0], self.hp_bound[1]],
        [-self.hp_bound[0], -self.hp_bound[1]],
    ])
    sensor_line = sensor_info - s_uav_pos_i
    s_uav_pos_i_m = np.zeros((self.hp_lidar_n, self.hp_dim)) + s_uav_pos_i  # uav pos reshape
    for oi in range(4):
        win_point = win_coord[oi]
        win_vector = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
        win_point_m = np.zeros((self.hp_lidar_n, self.hp_dim)) + win_point
        win_vector_m = np.zeros((self.hp_lidar_n, self.hp_dim)) + win_vector
        epsilon = 1e-6
        cross_win_sensor = np.cross(win_vector_m, sensor_line)
        cross_win_sensor_mask = (cross_win_sensor != 0) # sensor_line not parallel to win_vector
        ratio_winvec = np.cross((s_uav_pos_i_m - win_point_m), sensor_line) / (np.cross(win_vector_m, sensor_line) + epsilon)   # Possibility of intersection between a line and line segment
        ratio_sensor = np.cross((s_uav_pos_i_m - win_point_m), win_vector_m) / (np.cross(win_vector_m, sensor_line) + epsilon)  # Considering the length of the ray
        # 0<=ratio<=1: the intersection point is on win_vector_m; ration<0: on the opposite direction, ration>1: intersection point not on sensor line
        # 0<=um<=1: the intersection point is on the sensor line; um<0: on the opposite direction, um>1: intersection point not on sensor line
        winvec_sensor_mask = ((ratio_winvec >= 0) & (ratio_winvec <= 1) & (ratio_sensor >= 0) & (ratio_sensor <= 1))
        ratio_winvec = ratio_winvec.reshape(-1, 1)
        intersections = sensor_info.copy()
        mask = cross_win_sensor_mask * winvec_sensor_mask   # '*' is equal to '&' in this context

        intersections[mask] = (win_point_m + ratio_winvec * win_vector_m)[mask]
        sensor_info = intersections

    min_sensor_distance_res = np.linalg.norm(sensor_info - s_uav_pos_i_m, axis=1)

    # Range and direction construct the whole lidar information
    rela_goal_pos = self.s_goal - self.s_uav_pos[i]
    rot_mat = self.global_to_local(rela_goal_pos)
    trans_rot = np.dot(rot_mat, np.array([cos_rot, sin_rot]))
    if self.hp_coord_mode == 'local':
        return np.hstack([min_sensor_distance_res, trans_rot])
    else:
        return np.hstack([min_sensor_distance_res, np.array([cos_rot, sin_rot])])

# Breif:
#   Return the point cloud of a lidar mounted on uav_pos in the scene where a cirle centered at obs_pos with radius r
# Input:
#   uav_pos: position of the lidar
#   uav_sensor_pos: current point cloud (already add uav_pos)
#   obs_pos, r: position and radius of a circle
# Output:
#   possible_sensor_distance: norm of each point cloud line segment
#   possible_intersections: point cloud
def matrix_segment_circle_intersection(self, uav_pos, uav_sensor_pos, obs_pos, r):
    possible_sensor_distance = np.zeros(self.hp_lidar_n) + self.hp_lidar_range
    possible_intersections = uav_sensor_pos.copy()  # Use copy(), or else something wrong
    # expand vector to matrix
    uav_pos_m = np.zeros((self.hp_lidar_n, self.hp_dim)) + uav_pos   #p1m # uav_pos shape (1, hp_dim) --> (hp_n_sensors, hp_dim)
    uav_sensor_pos_m = uav_sensor_pos.copy() # p2m
    obs_pos_m = np.zeros((self.hp_lidar_n, self.hp_dim)) + obs_pos # p3m # obs_pos shape (1, hp_dim) --> (hp_n_sensors, hp_dim)

    # Solve the equation: norm(u \vec{a} - \vec{b}) = r, where u is a ratio, \vec{a} and \vec{b} are vectors starts at uav_pos, ends at a specific point of point cloud and the center of the obstacles respectively
    Am = np.linalg.norm(uav_sensor_pos_m - uav_pos_m, axis=1) ** 2
    Bm = 2 * ((uav_sensor_pos_m[:, 0] - uav_pos_m[:, 0]) * (uav_pos_m[:, 0] - obs_pos_m[:, 0]) + (uav_sensor_pos_m[:, 1] - uav_pos_m[:, 1]) * (uav_pos_m[:, 1] - obs_pos_m[:, 1]))
    Cm = np.linalg.norm(obs_pos_m, axis=1) ** 2 + np.linalg.norm(uav_pos_m, axis=1) ** 2 - 2 * np.sum(obs_pos_m * uav_pos_m, axis=1) - r ** 2
    delta = pow(Bm, 2) - 4 * Am * Cm
    for si, delta in enumerate(delta):
        u = None
        if delta < 0:
            continue
        elif delta == 0:
            u = -float(Bm[si]) / (2 * Am[si])
            if u < 0 or u > 1:
                continue
            else:
                u = u
        else:
            u1 = (-Bm[si] + math.sqrt(delta)) / (2 * Am[si])
            u2 = (-Bm[si] - math.sqrt(delta)) / (2 * Am[si])
            minu = min(u1, u2)
            maxu = max(u1, u2)
            if (minu >= 0 and minu <= 1) or (maxu >= 0 and maxu <= 1):
                if minu >= 0 and minu <= 1:
                    u = minu
                elif (maxu >= 0 and maxu <= 1):
                    u = maxu
            else:
                continue
        if u is not None:
            intersection = uav_pos + u * (uav_sensor_pos[si, :] - uav_pos)
            possible_intersections[si] = intersection
            possible_sensor_distance[si] = np.linalg.norm(u * (uav_sensor_pos_m[si] - uav_pos))
    return possible_sensor_distance, possible_intersections

def _vis_o_env_sensor(self):
    # visualize lidar as lines
    # if self.vis_init:
    #     self.vis_env = []
    #     for _ in range(self.hp_uav_n):
    #         lines = []
    #         for _ in range(self.hp_lidar_n):
    #             line = rendering.Line()
    #             self.vis_viewer.add_geom(line)
    #             lines.append(line)
    #         self.vis_env.append(lines)

    lidar_color1 = (216. / 255., 134. / 255., 59. / 255., 0.2)
    lidar_color2 = (109. / 255., 237. / 255., 253. / 255., 0.2)
    lidar_color3 = (222. / 255., 157. / 255., 172. / 255., 0.3)
    lidarline_color1 = (216. / 255., 134. / 255., 59. / 255., 1.)
    lidarline_color2 = (109. / 255., 237. / 255., 253. / 255., 1.)
    lidarline_color3 = (222. / 255., 157. / 255., 172. / 255., 1.)

    uav_sensor_point = []
    for i in range(self.hp_uav_n):  # self.hp_uav_n
        sensor_interpoint = []
        local_uav_direction = self.o_env_prev[i][-2:]
        rela_goal_pos = self.s_goal - self.s_uav_pos_prev[i]
        rot_inv_mat = self.local_to_global(rela_goal_pos)
        if self.hp_coord_mode == 'local':
            uav_direction = np.dot(rot_inv_mat, local_uav_direction)
        else:
            uav_direction = local_uav_direction
        cos_uav_direction = uav_direction[0]
        sin_uav_direction = uav_direction[1]
        if self.hp_lidar_angle_min == -np.pi:  # -self.hp_lidar_angle_max:
            theta = np.linspace(self.hp_lidar_angle_min, self.hp_lidar_angle_max, self.hp_lidar_n + 1)
            theta = theta[:self.hp_lidar_n]
        else:
            theta = np.linspace(self.hp_lidar_angle_min, self.hp_lidar_angle_max, self.hp_lidar_n)
        for j in range(self.hp_lidar_n):
            sensor_distance = self.o_env_prev[i][j]
            sensor_theta = theta[j]
            sensor_pos_x = self.s_uav_pos_prev[i][0] + sensor_distance * (
                        np.cos(sensor_theta) * cos_uav_direction - np.sin(sensor_theta) * sin_uav_direction)
            sensor_pos_y = self.s_uav_pos_prev[i][1] + sensor_distance * (
                        np.sin(sensor_theta) * cos_uav_direction + np.cos(sensor_theta) * sin_uav_direction)
            # self.vis_env[i][j].start = self.s_uav_pos_prev[i]
            # self.vis_env[i][j].end = [sensor_pos_x, sensor_pos_y]
            sensor_interpoint.append([sensor_pos_x, sensor_pos_y])
        uav_sensor_point.append(sensor_interpoint)

    # visualize lidar as polygons
    for _ in range(self.hp_uav_n):
        for j in range(self.hp_lidar_n):
            lidar = rendering.make_polygon(
                [uav_sensor_point[_][j], uav_sensor_point[_][(j + 1) % self.hp_lidar_n], self.s_uav_pos_prev[_]],
                filled=True)

            if _ == 0:
                lidar.set_color(*lidar_color1)
            elif _ == 1:
                lidar.set_color(*lidar_color2)
            else:
                lidar.set_color(*lidar_color3)
            self.vis_viewer.add_onetime(lidar)

        lidarline = rendering.make_polygon(uav_sensor_point[_], filled=False)

        if _ == 0:
            lidarline.set_color(*lidarline_color1)
        elif _ == 1:
            lidarline.set_color(*lidarline_color2)
        else:
            lidarline.set_color(*lidarline_color3)
        self.vis_viewer.add_onetime(lidarline)

# Partner

def _o_partner_pose(self):
    get = _get_o_partner_pose.__get__(self)

    if self.hp_uav_n > 1:
        one_uav_pos = np.ones((2, self.hp_dim))
        one_uav_pos[0, :] = -2 * self.hp_bound
        one_uav_pos[1, :] = 2 * self.hp_bound
        lim = np.hstack([one_uav_pos for _ in range(self.hp_uav_n - 1)])
    else:
        lim = np.zeros((2, 0))

    vis = _vis_o_partner_pose.__get__(self)
    return get, lim, vis

def _get_o_partner_pose(self, i):
    p = self.s_uav_pos[i]
    res = self.s_uav_pos - p
    res = np.delete(res, i, 0)

    # Represent the vectors in the local coordinate
    rela_goal_pos = self.s_goal - p
    rot_mat = self.global_to_local(rela_goal_pos)  # self.s_goal-p
    local_res = np.dot(rot_mat, res.T)  # res = [[x1,y1],[x2,y2]], need to be transposed

    if self.hp_coord_mode == 'local':
        return local_res.T  # transposed to original type
    else:
        return res

def _vis_o_partner_pose(self):
    if True:  # self.vis_init:
        partner_color = (25. / 255., 144. / 255., 2. / 255.)
        self.vis_partner = []
        for _ in range(self.hp_uav_n):
            lines = []
            for _ in range(self.hp_uav_n - 1):
                line = rendering.Line()
                line.set_color(*partner_color)
                # self.vis_viewer.add_geom(line)
                self.vis_viewer.add_onetime(line)
                lines.append(line)
            self.vis_partner.append(lines)

    for i in range(self.hp_uav_n):
        uav_rela_pos = self.o_partner_prev[i]
        rela_goal_pos = self.s_goal - self.s_uav_pos_prev[i]
        rot_inv_mat = self.local_to_global(rela_goal_pos)  # uav_rela_pos
        for j in range(self.hp_uav_n - 1):
            self.vis_partner[i][j].start = self.s_uav_pos_prev[i]
            local_uav_rela_pos = uav_rela_pos[j]
            global_uav_rela_pos = np.dot(rot_inv_mat, local_uav_rela_pos)
            if self.hp_coord_mode == 'local':
                self.vis_partner[i][j].end = self.s_uav_pos_prev[i] + global_uav_rela_pos
            else:
                self.vis_partner[i][j].end = self.s_uav_pos_prev[i] + local_uav_rela_pos

# Goal

def _o_goal_pose(self):
    get = _get_o_goal_pose.__get__(self)

    lim = np.ones((2, self.hp_dim))
    lim[0, :] = -2 * self.hp_bound
    lim[1, :] = 2 * self.hp_bound

    vis = _vis_o_goal_pose.__get__(self)
    return get, lim, vis

def _get_o_goal_pose(self, i):
    p = self.s_uav_pos[i]
    rela_goal_pos = self.s_goal - p
    rot_mat = self.global_to_local(rela_goal_pos)
    local_res = np.dot(rot_mat, rela_goal_pos)

    if self.hp_coord_mode == 'local':
        return local_res
    else:
        return rela_goal_pos

def _vis_o_goal_pose(self):
    if True:  # self.vis_init:
        goal_color = (255. / 255., 189. / 255., 0.)
        self.vis_goal = []
        for _ in range(self.hp_uav_n):
            line = rendering.Line()
            line.set_color(*goal_color)
            # self.vis_viewer.add_geom(line)
            self.vis_viewer.add_onetime(line)
            self.vis_goal.append(line)

    for i in range(self.hp_uav_n):
        self.vis_goal[i].start = self.s_uav_pos_prev[i]
        rela_goal_pos = self.s_goal - self.s_uav_pos_prev[i]
        rot_inv_mat = self.local_to_global(rela_goal_pos)  # local_o_goal_prev
        local_rela_goal_prev = self.o_goal_prev[i]
        global_o_goal_prev = np.dot(rot_inv_mat, local_rela_goal_prev)
        if self.hp_coord_mode == 'local':
            self.vis_goal[i].end = self.s_uav_pos_prev[i] + global_o_goal_prev
        else:
            self.vis_goal[i].end = self.s_uav_pos_prev[i] + local_rela_goal_prev

# Previous information

## No previous information
def _o_prev_none(self):
    get = _get_o_prev_none.__get__(self)

    lim = np.zeros((2, 0))

    vis = _vis_o_prev_none.__get__(self)
    return get, lim, vis

def _get_o_prev_none(self, i):

    return np.zeros(0)

def _vis_o_prev_none(self):

    return

## Previous velocity
def _o_prev_v(self):
    get = _get_o_prev_v.__get__(self)

    lim = np.ones((2, self.hp_dim))
    lim[0, :] = -self.hp_uav_v_max
    lim[1, :] = self.hp_uav_v_max

    vis = _vis_o_prev_v.__get__(self)

    return get, lim, vis

def _get_o_prev_v(self, i):
    s_goal = self.s_goal
    s_uav_pos = self.s_uav_pos[i]
    rela_goal_pos = s_goal - s_uav_pos
    rot_mat = self.global_to_local(rela_goal_pos)
    s_uav_v_prev = self.s_uav_v_prev[i].copy()
    local_uav_v_prev = np.dot(rot_mat, s_uav_v_prev)
    if self.hp_coord_mode == 'local':
        return local_uav_v_prev
    else:
        return s_uav_v_prev

def _vis_o_prev_v(self):
    if True:  # self.vis_init:
        prev_v_color = (0., 1., 0.)
        self.vis_prev_v = [rendering.Line() for _ in range(self.hp_uav_n)]
        for i in range(self.hp_uav_n):
            self.vis_prev_v[i].set_color(*prev_v_color)
            self.vis_prev_v[i].linewidth.stroke = 2
            # self.vis_viewer.add_geom(self.vis_prev_v[i])
            self.vis_viewer.add_onetime(self.vis_prev_v[i])

    for i in range(self.hp_uav_n):
        self.vis_prev_v[i].start = self.s_uav_pos_prev[i]
        local_uav_v_prev_prev = self.o_prev_prev[i]
        s_goal = self.s_goal
        s_uav_pos_prev = self.s_uav_pos_prev[i]
        rela_goal_pos = s_goal - s_uav_pos_prev
        rot_inv_mat = self.local_to_global(rela_goal_pos)
        global_uav_v_prev_prev = np.dot(rot_inv_mat, local_uav_v_prev_prev)
        if self.hp_coord_mode == 'local':
            self.vis_prev_v[i].end = self.s_uav_pos_prev[i] + global_uav_v_prev_prev
        else:
            self.vis_prev_v[i].end = self.s_uav_pos_prev[i] + local_uav_v_prev_prev
