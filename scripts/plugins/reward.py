import numpy as np

# Collision

def _get_rew_coll_interplote(self, action):
    for i in range(self.hp_interplote_times):
        tmp_dt = 1. * (i + 1) / self.hp_interplote_times * self.hp_dt
        uav_pos = self.get_s_uav_pos(action, tmp_dt)
        obs_pos = self.get_s_obs_pos(tmp_dt)
        self.var_coll = self.has_collision(uav_pos, obs_pos)
        if self.var_coll:
            break
    return self.hp_rew_coll if self.var_coll else 0.

# Goal

## Use dis_old - dis_new as metric
def _get_rew_goal_increment(self, action):
    if self.var_coll:
        return 0.
    p_old = np.average(self.s_uav_pos, axis=0)
    dis_old = np.linalg.norm(p_old - self.s_goal)
    for i in range(self.hp_interplote_times):
        tmp_dt = 1. * (i + 1) / self.hp_interplote_times * self.hp_dt
        uav_pos = self.get_s_uav_pos(action, tmp_dt)
        centroid = np.average(uav_pos, axis=0)
        dis_new = np.linalg.norm(centroid - self.s_goal)
        if dis_new <= self.hp_goal_r + self.hp_centroid_r:
            self.var_goal = True
            dis_new = 0.
        if self.var_goal:
            break
    rew = dis_old - dis_new
    rew += self.hp_rew_goal if self.var_goal else 0.
    return rew

# Formation

## Connectivity

### Penalize when distances are larger than dis_max
def _get_rew_formation_avgreladis_max(self, action):
    uav_pos = self.get_s_uav_pos(action, self.hp_dt)
    sum_dis = 0
    linknum = 0
    sum_cost_distance_limitation = 0.
    for i in range(uav_pos.shape[0]):
        for j in range(i + 1, uav_pos.shape[0]):
            dis = np.linalg.norm(self.s_uav_pos[i] - self.s_uav_pos[j])
            if dis > self.hp_uav_pref_dis_max:
                cost_i = abs(dis - self.hp_uav_pref_dis_max)
            else:   # Just avoid exceeding the limitation
                cost_i = 0.

            sum_cost_distance_limitation += cost_i
            linknum += 1

    cost_distance_limitation = sum_cost_distance_limitation / linknum
    return -cost_distance_limitation

### Penalize when distances are larger than dis_max or less than dis_min
def _get_rew_formation_avgreladis_interval(self, action):
    uav_pos = self.get_s_uav_pos(action, self.hp_dt)
    sum_dis = 0
    linknum = 0
    sum_cost_distance_limitation = 0.
    for i in range(uav_pos.shape[0]):
        for j in range(i + 1, uav_pos.shape[0]):
            dis = np.linalg.norm(self.s_uav_pos[i] - self.s_uav_pos[j])
            if dis > self.hp_uav_pref_dis_max:
                cost_i = abs(dis - self.hp_uav_pref_dis_max)
            elif dis < self.hp_uav_pref_dis_min:  # Want exactly the desirable distance
                cost_i = abs(dis - self.hp_uav_pref_dis_min)
            else:   # Just avoid exceeding the limitation
                cost_i = 0.

            sum_cost_distance_limitation += cost_i
            linknum += 1

    cost_distance_limitation = sum_cost_distance_limitation / linknum
    return -cost_distance_limitation

def _get_rew_formation_none(self, action):
   return 0.

# Preference

## Speed (v_max / v_pref)
def _get_rew_pref_v(self, action):
    _, v = self.get_s_uav_v(action, self.hp_dt)
    normv = np.linalg.norm(v, axis=1)
    mask = (normv > self.hp_uav_v_max)
    res = normv - self.hp_uav_v_max
    return -np.sum(res[mask]) / self.hp_uav_n

## Smoothness (measured by angle)
def _get_rew_vpenalty_vdifference(self, action):
    vprevprev = self.s_uav_v_prev
    vprev, _ = self.get_s_uav_v(action, self.hp_dt)
    lv1 = np.linalg.norm(vprev, axis=1)
    lv2 = np.linalg.norm(vprevprev, axis=1)
    if np.any(lv2):
        mask = (lv1 * lv2 > 0.)  # Avoid dividing zeros
        cos_angle = (vprev[:, 0] * vprevprev[:, 0] + vprev[:, 1] * vprevprev[:, 1])[mask] / (lv1 * lv2)[mask]
        cos_angle = np.clip(cos_angle, -1, 1)   # Why bother clipping cos to [-1, 1] ?
        angle = np.arccos(cos_angle)
        cost_vpenalty = sum(abs(angle))
    else:
        cost_vpenalty = 0.
    return -cost_vpenalty / self.hp_uav_n

# Efficiency

## A constant cost at each time step
def _get_rew_time_calsteps(self):
    cost_time = self.hp_dt
    return -cost_time
