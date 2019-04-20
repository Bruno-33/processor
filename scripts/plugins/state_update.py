import numpy as np

# Uav

## Input is velocity
def _model_normal_v(self):
    return _get_s_uav_pos_normal_v.__get__(self), \
           _get_s_uav_v_normal_v.__get__(self), \
           _get_s_uav_a_zero.__get__(self)

def _get_s_uav_pos_normal_v(self, v, t):
    mat_v, _ = _get_s_uav_v_normal_v(self, v, t)
    return self.s_uav_pos + mat_v * t

def _get_s_uav_v_normal_v(self, v, t):
    raw_mat_v = v.reshape((self.hp_uav_n, self.hp_dim)).copy()
    clip_mat_v = self.clip_to_max(v, self.hp_uav_v_max)
    return clip_mat_v, raw_mat_v

def _get_s_uav_a_zero(self, action, t):
    return np.zeros((self.hp_uav_n, self.hp_dim)), np.zeros((self.hp_uav_n, self.hp_dim))

## Input is acceleration
def _model_normal_a(self):
    return _get_s_uav_pos_normal_a.__get__(self), \
           _get_s_uav_v_normal_a.__get__(self), \
           _get_s_uav_a_normal_a.__get__(self)

def _get_s_uav_pos_normal_a(self, a, t):
    mat_a, _ = _get_s_uav_a_normal_a(self, a, t)
    mat_v, _ = _get_s_uav_v_normal_a(self, a, t)
    epsilon = np.zeros((self.hp_uav_n, self.hp_dim)) + 1e-6
    peak_t = (mat_v - self.s_uav_v_prev) / (mat_a + epsilon)
    assert np.any(peak_t[:, 0] - peak_t[:, 1] < 1e-3) and np.any(peak_t[:, 0] >= 0.)
    acc_t = peak_t[:, 0]    # Interval using the update rule: s' = s + vt + 1/2at**2
    uni_t = t - acc_t   # Interval using the update rule: s' = s + vt
    acc_offset = self.s_uav_v_prev * np.expand_dims(acc_t, axis=1)\
                + 1./2. * mat_a * np.expand_dims(acc_t ** 2, axis=1)
    uni_offset = mat_v * np.expand_dims(uni_t, axis=1)
    return self.s_uav_pos + acc_offset + uni_offset

def _get_s_uav_v_normal_a(self, a, t):
    clip_mat_a, raw_mat_a = _get_s_uav_a_normal_a(self, a, t)

    # Calculate the time t when norm(v+at) = v_max
    complex_peak_t = []
    for i in range(self.hp_uav_n):
        a_i = clip_mat_a[i]
        v_i = self.s_uav_v_prev[i]
        p = np.zeros(3)
        p[0] = np.dot(a_i, a_i)
        p[1] = 2. * np.dot(a_i, v_i)
        p[2] = np.dot(v_i, v_i) - self.hp_uav_v_max ** 2
        root = np.roots(p)
        if root.shape[0] == 0:  # No solution, e.g., a=0
            complex_peak_t.append(t * np.ones(2))
        else:
            complex_peak_t.append(root)
    complex_peak_t = np.array(complex_peak_t)  # dtype is complex128
    valid_mask = np.isreal(complex_peak_t) & (complex_peak_t.real >= 0.)    # Assume current velocity is leq than v_max, thus at most one postivie real solution
    peak_t = complex_peak_t.real    # Actually, with the assumption of leq v_max, all the solutions should be real

    # TODO: How did this method fail?
    # peak_t[~valid_mask] = np.inf
    # peak_t = np.min(peak_t, axis=1)
    # peak_t = np.clip(peak_t, a_max=t, a_min=None)
    peak_t[~valid_mask] = 0.
    peak_t = np.sum(peak_t, axis=1)
    peak_t = np.clip(peak_t, a_max=t, a_min=None)

    raw_mat_v = self.s_uav_v_prev + raw_mat_a * t
    clip_mat_v = self.s_uav_v_prev + clip_mat_a * np.expand_dims(peak_t, axis=1)
    return clip_mat_v, raw_mat_v

def _get_s_uav_a_normal_a(self, a, t):
    raw_mat_a = a.copy()
    raw_mat_a = raw_mat_a.reshape((self.hp_uav_n, self.hp_dim))
    clip_mat_a = self.clip_to_max(a, self.hp_uav_a_max)
    return clip_mat_a, raw_mat_a

## Input is velocity, with delay model v' = v * (1 - exp(-t/T))
def _model_delay_v(self):
    return _get_s_uav_pos_delay_v.__get__(self), \
           _get_s_uav_v_delay_v.__get__(self), \
           _get_s_uav_a_zero.__get__(self)

def _get_s_uav_pos_delay_v(self, v, t):
    mat_v = v.reshape((self.hp_uav_n, self.hp_dim))
    clip_mat_v, _ = _get_s_uav_v_delay_v(self, v, t)

    peak_t = -1. * self.hp_delay_t * np.log(1. - clip_mat_v / mat_v)
    assert np.any(peak_t[:, 0] - peak_t[:, 1] < 1e-3) and np.any(peak_t[:, 0] >= 0.)
    acc_t = peak_t[:, 0]    # Interval using the delay model update rule
    uni_t = t - acc_t   # Interval using the update rule: s' = s + vt
    acc_factor = (acc_t + self.hp_delay_t * np.exp(-1. * acc_t / self.hp_delay_t)) - self.hp_delay_t    # Remember to minus t=0
    acc_offset = mat_v * np.expand_dims(acc_factor, axis=1)
    uni_offset = clip_mat_v * np.expand_dims(uni_t, axis=1)

    return self.s_uav_pos + acc_offset + uni_offset

def _get_s_uav_v_delay_v(self, v, t):
    mat_v = v.reshape((self.hp_uav_n, self.hp_dim))
    raw_mat_v = mat_v * (1. - np.exp(-1. * t / self.hp_delay_t))    # The delay model

    # Compute the time t when the speed reach v_max
    acc_t = 1. - (self.hp_uav_v_max / np.linalg.norm(mat_v, axis=1))
    reachable_mask = acc_t > 0.
    acc_t[~reachable_mask] = t  # Input speed <= v_max
    acc_t[reachable_mask] = -1. * self.hp_delay_t * np.log(acc_t[reachable_mask])   # Input speed > v_max
    acc_t = np.clip(acc_t, a_min=None, a_max=t)
    factor = (1. - np.exp(-1. * acc_t / self.hp_delay_t))
    clip_mat_v = mat_v * np.expand_dims(factor, axis=1)

    return clip_mat_v, raw_mat_v

# Obstacle

def _get_s_obs_pos_uniform(self, t):
    return self.s_obs_pos + self.s_obs_v * t
