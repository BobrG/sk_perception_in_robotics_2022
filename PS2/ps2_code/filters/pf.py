"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)
        # TODO add here specific class variables for the PF
        
        self.num_particles = num_particles
        self.X = np.zeros((num_particles, 3))
        #self.particles = np.repeat(initial_state.mu, repeats=self.num_particles, axis=1).T
        self.weights = np.zeros((num_particles, 1))
        for i in range(len(initial_state.mu)):
            sample = gaussian.rvs(initial_state.mu[i], initial_state.Sigma[i, i], size=num_particles)
            self.X[:, i] = sample

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        self.X = np.array([sample_from_odometry(self.X[i], u, self._alphas) for i in range(self.num_particles)])
        sample_gauss = get_gaussian_statistics(self.X)
        
        self._state_bar.mu = sample_gauss.mu
        if self.mu[-1] < -np.pi or self.mu[-1] > np.pi:
            self.mu[-1] = wrap_angle(self.mu[-1])

        self._state_bar.Sigma = sample_gauss.Sigma

    def update(self, z):
        # TODO implement correction step
        for i in range(self.num_particles):
            h_x_m = get_observation(self.X[i], z[1])[0]
            self.weights[i] = gaussian().pdf(wrap_angle(z[0] - h_x_m) / np.sqrt(self._Q)) / np.sqrt(self._Q)
        self.weights /= np.sum(self.weights)
        self.resampling()

        sample_gauss = get_gaussian_statistics(self.X)
        self._state.mu = sample_gauss.mu
        if self._state.mu[-1] < -np.pi or self._state.mu[-1] > np.pi:
            self._state.mu[-1] = wrap_angle(self._state.mu[-1])

        self._state.Sigma = sample_gauss.Sigma
        
    def resampling(self):
        # low variance sampling
        r = uniform(0, 1/self.num_particles)
        c = self.weights[0][0]
        i = 0

        new_particles = np.zeros((self.num_particles, 3))
        new_weights = np.zeros((self.num_particles, 1))

        for m in range(0, self.num_particles):
            a = r + (m/self.num_particles)
            while a > c:
                c += self.weights[i][0]
                i += 1
            new_particles[m] = self.X[i-1]
            new_weights[m] = 1/self.num_particles 
        self.X = new_particles
        self.weights = new_weights 
