import numpy as np

# Uav and goal position
def _get_init_uav_pos_and_goal_diag_normal(self):
    return _get_init_uav_pos_and_goal_diag(self, uav_pos_range=np.array([2. / 3., 9. / 10.]), goal_pos_range=np.array([2. / 3., 9. / 10.]))

def _get_init_uav_pos_and_goal_diag_safer(self):
    return _get_init_uav_pos_and_goal_diag(self, uav_pos_range=np.array([2. / 3., 3. / 4.]), goal_pos_range=np.array([7. / 12., 19. / 24.]))

def _get_init_uav_pos_and_goal_diag(self, uav_pos_range, goal_pos_range):
    # Template
    if self.hp_uav_n == 3:
        template = np.array([[0., 2.], [-2., -2.], [2., -2.]]) / 4.
    elif self.hp_uav_n == 1:
        template = self.np_random.rand(self.hp_uav_n * self.hp_dim).reshape((self.hp_uav_n, -1)) / 4.
    else:   # Circle template
        theta = np.linspace(0, 2 * np.pi, self.hp_uav_n + 1)
        theta = theta[:-1]
        template = np.zeros((self.hp_uav_n, self.hp_dim))
        template[:, 0] = np.cos(theta) * 2. * (self.hp_uav_n / 3.)
        template[:, 1] = np.sin(theta) * 2. * (self.hp_uav_n / 3.)
        template = template / (4.*(3./self.hp_uav_n) * 2.)

    # Offset
    diag_pos = np.array([[-self.hp_bound[0], -self.hp_bound[1]], [-self.hp_bound[0], self.hp_bound[1]], \
                         [self.hp_bound[0], self.hp_bound[1]], [self.hp_bound[0], -self.hp_bound[1]]])
    uav_offset_i = self.np_random.randint(4)
    uav_offset = np.array([self.np_random.uniform(*(diag_pos[uav_offset_i, 0] * uav_pos_range)),
                           self.np_random.uniform(*(diag_pos[uav_offset_i, 1] * uav_pos_range))])
    goal_offset_i = (uav_offset_i + self.np_random.randint(1, 4)) % 4   # Guarantee different corner
    goal_offset = np.array([self.np_random.uniform(*(diag_pos[goal_offset_i, 0] * goal_pos_range)),
                            self.np_random.uniform(*(diag_pos[goal_offset_i, 1] * goal_pos_range))])

    return template + uav_offset, goal_offset

def _get_init_uav_pos_and_goal_real(self):
    bottom_range = np.array([-9./10., -7./8.])

    if self.hp_uav_n == 3:
        template = np.array([[0., 1.4], [-0.8, 0.], [0.8, 0.]])
    else:
        template = self.np_random.rand(self.hp_uav_n * self.hp_dim).reshape((self.hp_uav_n, -1))

    uav_offset = np.zeros(self.hp_dim)
    for i, lim in enumerate(self.hp_bound):
        if i == len(self.hp_bound)-1:
            uav_offset[i] = self.np_random.uniform(*(lim * bottom_range))
        else:
            uav_offset[i] = self.np_random.uniform(-0.2, .2)

    goal_offset = np.zeros(self.hp_dim)
    for i, lim in enumerate(self.hp_bound):
        if i == len(self.hp_bound)-1:
            top_range = np.array([0., .5])
            goal_offset[i] = self.np_random.uniform(*(2.5 + top_range))
        else:
            goal_offset[i] = self.np_random.uniform(0.2, 1.2)

    return template + uav_offset, goal_offset

def _get_init_uav_pos_and_goal_manual(self):
    bottom_range = np.array([-9./10., -7./8.])

    if self.hp_uav_n == 3:
        template = np.array([[0., 1.4], [-0.8, 0.], [0.8, 0.]])
    else:
        template = self.np_random.rand(self.hp_uav_n * self.hp_dim).reshape((self.hp_uav_n, -1))

    uav_offset = np.zeros(self.hp_dim)
    for i, lim in enumerate(self.hp_bound):
        if i == len(self.hp_bound)-1:
            uav_offset[i] = self.np_random.uniform(*(lim * bottom_range))
        else:
            uav_offset[i] = self.np_random.uniform(-0.2, .2)

    goal_offset = np.array([0., 2.5])

    return template + uav_offset, goal_offset

# Obstacle

## Position and radius

def _get_init_obs_pos_and_r_dynamic(self):
    n_obs = self.np_random.randint(*self.hp_obs_n)
    safe_range = 5. / 6.
    success = False     # Whether successfully initialize all obstacles
    while True:
        res_pos = np.zeros((n_obs, self.hp_dim))
        res_r = np.zeros(n_obs)
        for j in range(n_obs):
            counter = 0     # How many trials have been taken for the current obstacle
            valid = False
            pos = np.zeros(self.hp_dim)
            r = np.zeros(1)
            while(counter < self.hp_trial_max and not valid):  # Haven't reached the trial threshold and not valid
                for i, lim in enumerate(self.hp_bound):
                    ind = self.np_random.randint(2)
                    if i == ind:
                        pos[i] = self.np_random.uniform(-lim, lim)
                    else:
                        pos[i] = self.np_random.uniform(-lim * safe_range, lim * safe_range)
                r = self.np_random.uniform(*self.hp_obs_r)
                valid = self.is_setup_valid_obs(pos, r, res_pos[:j], res_r[:j])
                counter += 1
            if valid:
                res_pos[j] = pos
                res_r[j] = r
                if j == n_obs-1:
                    success = True
            else:
                break
        if success:     # Only when all obstacles are initialized successfully can the loop be broken
            break
    return res_pos, res_r

def _get_init_obs_pos_and_r_real(self):
    n_obs = self.np_random.randint(*self.hp_obs_n)
    success = False     # Whether successfully initialize all obstacles
    while True:
        res_pos = np.zeros((n_obs, self.hp_dim))
        res_r = np.zeros(n_obs)
        for j in range(n_obs):
            counter = 0     # How many trials have been taken for the current obstacle
            valid = False
            pos = np.zeros(self.hp_dim)
            r = np.zeros(1)
            while(counter < self.hp_trial_max and not valid):  # Haven't reached the trial threshold and not valid
                for i, lim in enumerate(self.hp_bound):
                    if i == 0:
                        pos[i] = self.np_random.uniform(-lim, lim)
                    else:
                        pos[i] = self.np_random.uniform(-lim * 7./12., lim * 5./6.)
                r = self.np_random.uniform(*self.hp_obs_r)
                valid = self.is_setup_valid_obs(pos, r, res_pos[:j], res_r[:j])
                counter += 1
            if valid:
                res_pos[j] = pos
                res_r[j] = r
                if j == n_obs-1:
                    success = True
            else:
                break
        if success:     # Only when all obstacles are initialized successfully can the loop be broken
            break
    return res_pos, res_r

def _get_init_obs_pos_and_r_manual(self):
    res_pos = np.array([
        # Noisy obstacles around the boundary
        [-3.1, -3.],
        [-3.4, -1.5],
        [-3.3, -0],
        [-3.1, 3.],
        [-3.2, 1.5],
        [-3.5, 0],
        [-2.9, -2.],
        [-2.8, -2.5],
        [-2.9, -1],
        [-2.9, 4.],
        [-2.9, 2.5],
        [-2.9, 1],

        [3.1, -3.],
        [3.4, -1.5],
        [3.3, -0],
        [3.1, 3.],
        [3.2, 1.5],
        [3.5, 0],
        [2.9, -2.],
        [2.8, -2.5],
        [2.9, -1],
        [2.9, 4.],
        [2.9, 2.5],
        [2.9, 1],

        [-3., 4.3],
        [-2., 4.4],
        [-1., 4.5],
        [0., 4.2],
        [1., 4.3],
        [2., 4.4],
        [3., 4.5],
        # Valid obstacles
        [-1., -0.8],
        [1.5, -0.8],
        [0., 0.7],
        ])
    res_r = self.np_random.uniform(*self.hp_obs_r, res_pos.shape[0])
    return res_pos, res_r

## Velocity

def _get_init_obs_v_zero(self):
    n_obs = self.s_obs_pos.shape[0]
    return np.zeros((n_obs, self.hp_dim))
