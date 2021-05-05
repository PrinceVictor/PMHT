import numpy as np
from scipy.stats import poisson, norm

from .kalman import *

def compute_detection_prob(meas_size, target_size):
    return 1.0 if meas_size>= target_size else meas_size/target_size

def compute_norm_prob(z, y, R):
    z = np.mat(z)
    y = np.mat(y)
    R = np.mat(R)
    return 1/(np.power(2*np.pi, 2)*np.linalg.det(R))*np.exp(-0.5*(z-y).T*R.I*(z-y))

def compute_poisson_prob(expect, k):
    return poisson.pmf(k, expect)


class PMHT:
    def __init__(self, times):
        print("Construct PMHT!")
        self.noise_expected = 10
        self.meas_frequency = 0.1
        self.delta_t = 1/self.meas_frequency

        self.target_state = [None]*times
        self.P = [None]*times
        self.Q = get_process_noise_matrix(self.delta_t, sigma=0.01)
        self.R = get_measurement_noise_matrix(sigma=100)
    
    def pmht_init(self, measurements):
        print("PMHT target init!")
        
        target_state = np.mat(np.zeros((measurements.shape[0], 4, 1), dtype=np.float))
        self.P[0] = np.mat(np.zeros((measurements.shape[0], 4, 4), dtype=np.float))
        for index, meas in enumerate(measurements):
            target_state[index, 0] = meas[0]
            target_state[index, 2] = meas[1]
        
        self.target_state[0] = target_state

        
    def run(self, t_idx, measurements):
        print("Runing PMHT")
        
        print(f"t index: {t_idx}")
        print(measurements.shape)
        # print(measurements)

        if t_idx == 0:
            self.pmht_init(measurements)
        else:
            x_predict, P_predict = state_predict(self.target_state, self.P, self.Q, self.delta_t)

            self.calculate_weight_s(t_idx, x_predict, measurements)
    

    def calculate_weight_s(self, t_idx, x_predict, measurements):
        print("Target update!")
        
        pi_s = get_prior_prob(t_idx, measurements)
        for idx_r, per_meas in enumerate(measurements):

            w_lr_list = []
            for idx_l, per_target in enumerate(x_predict):
                compute_norm_prob(z, y, R)

    
    def get_prior_prob(self, t, measurements):
        print("Compute prior probabilities!")

        meas_size = measurements.shape[0]
        target_size = self.target[t-1].shape[0]
        expect_mu = self.noise_expected

        pd = get_detection_prob(meas_size, target_size)
        p_meas_size = compute_prior_prob(expect_mu, meas_size)
        p_meas_tag = compute_prior_prob(expect_mu, np.max((meas_size-target_size, 0)))
        
        pi_s = pd/meas_size*p_meas_tag/(pd*p_meas_tag + (1-pd)*p_meas_size)
        
        return pi_s

        
        

            
            
