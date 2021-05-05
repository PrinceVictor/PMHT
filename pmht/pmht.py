import numpy as np
from scipy.stats import poisson

def compute_detection_prob(meas_size, target_size):
    return 1.0 if meas_size>= target_size else meas_size/target_size

def compute_poisson_prob(expect, k):
    return poisson.pmf(k, expect)

def compute_prior_prob(meas_size, target_size, expect_mu):
    pd = get_detection_prob(meas_size, target_size)
    p_meas_size = compute_prior_prob(expect_mu, meas_size)
    p_meas_tag = compute_prior_prob(expect_mu, np.max((meas_size-target_size, 0)))
    return pd/meas_size*p_meas_tag/(pd*p_meas_tag + (1-pd)*p_meas_size)

class PMHT:
    def __init__(self, times):
        print("Construct PMHT!")
        self.noise_expected = 10
        self.target = [None]*times
    
    def target_init(self, measurements):
        print("Target init!")
        
        target_state = np.zeros((measurements.shape[0], 4, 1), 
                                dtype=np.float)
        for index, meas in enumerate(measurements):
            target_state[index, 0] = meas[0]
            target_state[index, 2] = meas[1]

    def get_prior_prob(self, t, measurements):
        print("Compute prior probabilities!")

        pi_s = compute_prior_prob(measurements.shape[0], 
                                  self.target[t-1].shape[0],
                                  self.noise_expected)

        # for idx_target, per_target in enumerate(self.target[t-1]):
        #     for idx_meas, per_meas in enumerate(measurements):
        return pi_s



    def target_update(self, t_idx, measurements):
        print("Target update!")
        
        pi_s = get_prior_prob(t_idx, measurements)



    def run(self, t_idx, measurements):
        print("Runing PMHT")
        
        print(t_idx)
        print(measurements.shape)
        # print(measurements)

        if t_idx == 0:
            self.target_init(measurements)
        else:
            self.target_update(t_idx, measurements)

        
        

            
            
