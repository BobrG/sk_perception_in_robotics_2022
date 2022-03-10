"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian

class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, verbose=False, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        self.verbose = verbose         
        self.graph = mrob.FGraph()

        self.landmarks_observed = {}
        self.nt = self.graph.add_node_pose_2d(initial_state.mu)
        self.xt = np.array(self.graph.get_estimated_state()[0])
        self.curr_lndmark = np.zeros(2)
        self.graph.add_factor_1pose_2d(initial_state.mu, self.nt, inv(initial_state.Sigma))
        if self.verbose:
            print('TASK 1.A:')
            print('------initial-state------')
            print('-------------------------')
            self.graph.print(True)
            print('-------------------------')

    def predict(self, u):
        if self.verbose:
            print('TASK 1.B:')
            print('------before-action------')
            print('-----estimated-state-----')
            print('-------------------------')
            print(self.graph.get_estimated_state())
            print('-------------------------')

        n_new = self.graph.add_node_pose_2d(np.zeros(3))
        _, V = state_jacobian(self.xt, np.array(u))
        W_u = inv(V@get_motion_noise_covariance(u, self.alphas)@V.T)
        self.graph.add_factor_2poses_2d_odom(u, self.nt, n_new, W_u)
        self.nt = n_new
        self.xt = self.graph.get_estimated_state()[-1]
        if self.verbose:
            print('-----estimated-state-----')
            print('-------------------------')
            print(self.graph.get_estimated_state())
            print('-------------------------')        

    def update(self, z):
        W_z = inv(self.Q)
        for (rng, brng, lid) in z:
            if lid in self.landmarks_observed.keys():
                initializeLandmark = False
            else:
                node_id = self.graph.add_node_landmark_2d(np.zeros(2))
                initializeLandmark = True
                self.landmarks_observed[lid] = node_id
            
            self.graph.add_factor_1pose_1landmark_2d(np.array([rng, brng]).reshape(2, 1), self.nt, 
                                                     self.landmarks_observed[lid], W_z, initializeLandmark)

        if self.verbose:
            print('TASK 1.C:')
            print('-after-observations-state-')
            print('--------------------------')
            out = 'START'
            print(self.graph.get_estimated_state())
            for x in self.graph.get_estimated_state():
                if len(x) == 3:
                    out += '-->Node ' + str(x.T)
                elif len(x) == 2:
                    out += 'Landmark ' + str(x.T) + '&'
            out += '-->END'
            print(out)
            print('--------------------------')

    def solve(self):
        self.graph.solve()
        if self.verbose:
            print('TASK 1.D')
            print('-full-graph-after-solve-')
            print('------------------------')
            self.graph.print(True)
            print('------------------------')

    def error(self):
        err = self.graph.chi2()
        print('ERROR:', err)

        return err

    def adj_matrix(self):
        return self.graph.get_adjacency_matrix()

    def inform_matrix(self):
        return self.graph.get_information_matrix()
