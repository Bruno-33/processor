import numpy as np

# Return rotation matrix of \pi/2-\theta counter-clockwise rotation, where \theta is the angle between the input vector and the vector [1, 0]
def _rot_mat_half_pi_minus_theta_counter_clockwise(rela_goal_pos):
    cos_theta = rela_goal_pos[1] / np.linalg.norm(rela_goal_pos)
    sin_theta = -rela_goal_pos[0] / np.linalg.norm(rela_goal_pos)
    mat = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    return mat

# Return rotation matrix of \pi/2-\theta clockwise rotation
def _rot_mat_half_pi_minus_theta_clockwise(rela_goal_pos):
    mat = _rot_mat_half_pi_minus_theta_counter_clockwise(rela_goal_pos)
    return np.linalg.inv(mat)

def _rot_mat_theta_clockwise(theta):
    mat = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])
    return mat
