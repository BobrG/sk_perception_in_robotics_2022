"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

def get_G(state, u):
    x, y, theta = state
    drot1, dtran, drot2 = u
    angle = theta + drot1
    if angle < -np.pi:
        angle = wrap_angle(angle)
    if angle > np.pi:
        angle = wrap_angle(angle)
    return np.array([[1, 0, -dtran*np.sin(angle)], 
                     [0, 1, dtran*np.cos(angle)], 
                     [0, 0, 1]])

def get_V(state, u):
    x, y, theta = state
    drot1, dtran, drot2 = u
    angle = theta + drot1

    if angle < -np.pi:
        angle = wrap_angle(angle)
    if angle > np.pi:
        angle = wrap_angle(angle)

    return np.array([[-dtran*np.sin(angle), np.cos(angle), 0],
                     [dtran*np.cos(angle), np.sin(angle), 0], 
                     [1, 0, 1]])

def get_H(state, map_vec):
    dx, dy = map_vec[0] - state[0], map_vec[1] - state[1]
    q = np.power(dx, 2) + np.power(dy, 2)
    return np.asarray([(map_vec[1] - state[1])/q, -(map_vec[0] - state[0])/q, -1])


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        # prerequisites
        G = get_G(self.mu, u)
        V = get_V(self.mu, u)
        self._state_bar.mu = get_prediction(self.mu, u)[np.newaxis].T
        self._state_bar.Sigma = G@self.Sigma@G.T + V@get_motion_noise_covariance(u, self._alphas)@V.T
        if self.mu[-1] < -np.pi or self.mu[-1] > np.pi:
            self.mu[-1] = wrap_angle(self.mu[-1])

    def update(self, z):
        # TODO implement correction step
        
        lnd_id = int(z[1])
        map_vec = [self._field_map.landmarks_poses_x[lnd_id], self._field_map.landmarks_poses_y[lnd_id]]
        H = get_H(self.mu, map_vec)

        z_bar = get_expected_observation(self.mu_bar, z[1])[0]
        S = H@self.Sigma_bar@H[np.newaxis].T + self._Q
        K = ((self.Sigma_bar@H[np.newaxis].T)/(S[0]))
        
        z_diff = (z[0] - z_bar)
        if z_diff < -np.pi or z_diff > np.pi:
            z_diff = wrap_angle(z_diff)
        self._state_bar.mu += K*z_diff
        self._state_bar.Sigma = (np.eye(self.Sigma.shape[-1]) - K@H[np.newaxis])@self.Sigma_bar
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
        if self._state.mu[-1] < -np.pi or self._state.mu[-1] > np.pi:
            self.mu[-1] = wrap_angle(self.mu[-1])

