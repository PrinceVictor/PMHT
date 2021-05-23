import numpy as np

from numpy.linalg import inv, det
from scipy.stats import poisson, norm

from .kalman import *

def compute_detection_prob(meas_size, target_size):
    return 1.0 if meas_size>= target_size else meas_size/target_size

def compute_norm_prob(z, y, R):

    return 1/(2*np.pi*(det(R)**0.5))*np.exp(np.array(-0.5*(z-y).T@inv(R)@(z-y))[0][0])

def compute_poisson_prob(expect, k):
    return poisson.pmf(k, expect)


class PMHT:
    def __init__(self, times, batch_T=1, noise_expected=10, sample_T=5):
        print("Construct PMHT!")
        self.batch_Tg = batch_T
        self.batch_Tb = self.batch_Tg + 2
        self.noise_expected = noise_expected
        self.delta_t = sample_T

        self.target_state = [None]*times
        self.meas_buff = []
        self.t_buff = []
        self.P = [None]*times
        self.Q = np.round(get_process_noise_matrix(self.delta_t, sigma=1))
        self.R = get_measurement_noise_matrix(sigma=500)
        self.H = measurement_matrix()
        self.pmht_init_flag = True
    
    def meas_manage(self, t_idx, meas):
        
        if len(self.meas_buff) >= self.batch_Tb:
            for i in range(self.batch_Tg):
                self.meas_buff.pop(0)
                self.t_buff.pop(0)
        self.meas_buff.append(meas)
        self.t_buff.append(t_idx)
        
        return len(self.meas_buff) >= self.batch_Tb
    
    def pmht_init(self, target_prior):
        print("PMHT target init!")
        
        target_state = np.zeros((target_prior.shape[0], 4, 1), dtype=np.float)
        self.P[0] = np.zeros((target_prior.shape[0], 4, 4), dtype=np.float)
        
        for index, per_tgt_prior in enumerate(target_prior):
            target_state[index, 0, 0] = per_tgt_prior[0]
            target_state[index, 1, 0] = per_tgt_prior[1]
            target_state[index, 2, 0] = per_tgt_prior[3]
            target_state[index, 3, 0] = per_tgt_prior[4]
        self.target_state[0] = target_state

        for t_idx in range(1, self.batch_Tg):    
            for idx in range(len(self.target_state[t_idx-1])):
                x = self.target_state[t_idx-1][idx]
                P = self.P[t_idx-1][idx]
                x, P = state_predict(x, P, self.Q, self.delta_t)

                self.target_state[t_idx] = x
                self.P[t_idx] = P

        self.pmht_init_flag = False
        
    def run(self, t_idx, measurements):
        print(f"Runing PMHT T:{t_idx}")

        meas_flag = self.meas_manage(t_idx, measurements)
            
        if meas_flag:
            x_predicts = []
            P_predicts = []
            for idx in range(len(self.target_state[t_idx-1])):
                x = self.target_state[t_idx-1][idx]
                p = self.P[t_idx-1][idx]
                x, p = state_predict(x, p, self.Q, self.delta_t)
                x_predicts.append(x)
                P_predicts.append(p)

            w_nsr = self.calculate_weight_s(t_idx, x_predicts, P_predicts, measurements)
            zs, Rs = self.calculate_measures_and_covariance(w_nsr, measurements)
            x_est, P_est = self.em_iteration(t_idx, x_predicts, P_predicts, zs, Rs)

            self.target_state[t_idx] = x_est
            self.P[t_idx] = P_est
    
    def get_track_info(self):
        return self.target_state

    def em_iteration(self, t_idx, x_predicts, P_predicts, zs, Rs):

        x_est = np.zeros((len(x_predicts), 4, 1), dtype=np.float)
        P_est = np.zeros((len(x_predicts), 4, 4), dtype=np.float)
        
        for idx_s in range(len(x_predicts)):

            while True:
                x = x_predicts[idx_s]
                P = P_predicts[idx_s]
                z = zs[idx_s]
                R = Rs[idx_s]

                if z is not None:
                
                    x1, P1 = state_update(x, P, z, R)
                    cost = (x1-x).T @ inv(self.Q) @ (x1-x)

                    x_predicts[idx_s] = x1
                    P_predicts[idx_s] = P1

                    if cost <= 0.01:
                        x_est[idx_s] = x1
                        P_est[idx_s] = P1
                        break
                else:
                    x_est[idx_s] = x
                    P_est[idx_s] = P
                    break
        
        return x_est, P_est

    def calculate_measures_and_covariance(self, w_nsr, measurements):
        print("synthetic measurements")

        zs = [None]*len(w_nsr)
        Rs = [None]*len(w_nsr)
        for idx_s, wns in enumerate(w_nsr):

            temp_z = np.mat(np.zeros((2, 1), dtype=np.float))
            for idx_r, wnsr in enumerate(wns):
                temp_z += wnsr*measurements[idx_r]
            
            sum_wns = np.sum(wns)

            if sum_wns != 0:
                temp_z = temp_z/sum_wns
                temp_R = self.R/sum_wns

                zs[idx_s] = temp_z
                Rs[idx_s] = temp_R
        
        return zs, Rs

    def calculate_weight_s(self, t_idx, x_predicts, P_predicts, measurements):
        print("Target update!")
        
        pi_s = self.get_prior_prob(t_idx, measurements)

        w_nsr_list = []
        for idx_r, z in enumerate(measurements):

            w_sr_list = []
            for idx_s, x in enumerate(x_predicts):
                
                w_sr_num = pi_s*compute_norm_prob(z, self.H@x, self.R)
                w_sr_list.append(w_sr_num)

            w_sr_den = np.sum(w_sr_list)
            if w_sr_den != 0:
                w_sr_list = w_sr_list/w_sr_den
            
            w_nsr_list.append(w_sr_list)
        
        w_nsr = []
        for idx_s in range(len(x_predicts)):
            w_sr = []
            for idx_r in range(len(measurements)):
                w_sr.append(w_nsr_list[idx_r][idx_s])
            w_nsr.append(w_sr)

        return w_nsr
    
    def get_prior_prob(self, t, measurements):
        print("Compute prior probabilities!")

        meas_size = measurements.shape[0]
        target_size = self.target_state[t-1].shape[0]
        expect_mu = self.noise_expected

        pd = compute_detection_prob(meas_size, target_size)
        p_meas_size = compute_poisson_prob(expect_mu, meas_size)
        p_meas_tag = compute_poisson_prob(expect_mu, np.max((meas_size-target_size, 0)))
        
        pi_s = pd/meas_size*p_meas_tag/(pd*p_meas_tag + (1-pd)*p_meas_size)
        
        return pi_s

        
        

            
            
