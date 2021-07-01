import numpy as np

from numpy.linalg import inv, det
from scipy.stats import poisson, norm

from .kalman import *
from pmht.track import MOT

def compute_detection_prob(meas_size, target_size):
    return 1.0 if meas_size>= target_size else meas_size/target_size

def compute_norm_prob(z, y, R):
    return 1/(2*np.pi*(det(R)**0.5))*np.exp(np.array(-0.5*(z-y).T@inv(R)@(z-y))[0][0])

def compute_poisson_prob(expect, k):
    return poisson.pmf(k, expect)


class PMHT:
    def __init__(self, times, batch_T=1, noise_expected=10, sample_T=5, meas_sigma=10):
        print("Construct PMHT!")
        self.batch_Tg = batch_T
        self.batch_buff = 2
        self.batch_Tb = self.batch_Tg + self.batch_buff
        self.noise_expected = noise_expected
        self.delta_t = sample_T
        self.times = times-self.batch_buff
        self.target_state = [None]* (self.times - self.times%self.batch_Tg)
        self.em_iteration_times = 0
        self.cost = []
        self.meas_buff = []
        self.t_buff = []
        self.P = [None]*(self.times - self.times%self.batch_Tg)
        self.Q = np.round(get_process_noise_matrix(self.delta_t, sigma=0.85))
        self.R = get_measurement_noise_matrix(sigma=meas_sigma)
        self.H = measurement_matrix()
        
        self.pmht_init_flag = False
        self.targets = []

        self.mot_track = MOT(times=times, 
                             delta_t=sample_T, keep_T=3,
                             meas_sigma=meas_sigma)
    
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
        
        xs = np.zeros((target_prior.shape[0], 4, 1), dtype=np.float)
        Ps = np.zeros((target_prior.shape[0], 4, 4), dtype=np.float)
        
        for index, per_tgt_prior in enumerate(target_prior):
            xs[index, 0, 0] = per_tgt_prior[0] + np.random.normal(0, 100, 1)
            xs[index, 1, 0] = per_tgt_prior[1] + np.random.normal(0, 5, 1)
            xs[index, 2, 0] = per_tgt_prior[3] + np.random.normal(0, 100, 1)
            xs[index, 3, 0] = per_tgt_prior[4] + np.random.normal(0, 5, 1)
        self.target_state[0] = xs
        self.P[0] = Ps
        
        for t_idx in range(1, self.batch_Tg):
            xs = np.zeros((target_prior.shape[0], 4, 1), dtype=np.float)
            Ps = np.zeros((target_prior.shape[0], 4, 4), dtype=np.float)    
            
            for idx in range(len(xs)):
                x = self.target_state[t_idx-1][idx]
                P = self.P[t_idx-1][idx]

                x, P = state_predict(x, P, self.Q, self.delta_t)
                
                xs[idx] = x
                Ps[idx] = P

            self.target_state[t_idx] = xs
            self.P[t_idx] = Ps
        self.pmht_init_flag = False
        
    def run(self, t_idx, measurements):
        print(f"Runing PMHT T:{t_idx}")
        if self.pmht_run_flag:
            meas_flag = self.meas_manage(t_idx, measurements)

        if not self.pmht_run_flag:
            self.mot_track(t_idx, measurements)
            targets_list = self.get_targets(t_idx, measurements)
            if len(targets_list):
                self.pmht_run_flag = True

        if meas_flag:        
            self.em_iteration_times = 0
            self.em_iteration()
            self.pmht_run_flag = False
            
    
    def get_track_info(self):
        return self.target_state
    
    def em_iteration(self):
        print("EM iteration!")

        iteration_flag = True

        while iteration_flag:
            x_t_l, P_t_l = self.target_predict()
            w_tnsr = self.calculate_weight_s(x_t_l)
            zts, Rts = self.calculate_measures_and_covariance(w_tnsr)
            x_est_l, P_est_l = self.batch_filter(x_t_l, P_t_l, zts, Rts)
            iteration_flag = self.calculate_cost(x_est_l)
            self.update_info(x_est_l, P_est_l)
    
    def target_predict(self):
        if self.em_iteration_times == 0:
            self.cost = [None]*len(self.target_state[self.t_buff[0]-1])

        x_t_l = [self.target_state[self.t_buff[0]-1]]
        P_t_l = [self.P[self.t_buff[0]-1]]
        for idt in range(self.batch_Tg):
            t_id = self.t_buff[idt]

            x_predicts = []
            P_predicts = []
            for idx in range(len(x_t_l[-1])):
                if self.cost[idx] is not None and self.cost[idx] < 0.01:
                    x = self.target_state[t_id][idx]
                    p = self.P[t_id][idx]
                else:    
                    x = x_t_l[-1][idx]
                    p = P_t_l[-1][idx]
                    x, p = state_predict(x, p, self.Q, self.delta_t)
                x_predicts.append(x)
                P_predicts.append(p)
            
            x_t_l.append(x_predicts)
            P_t_l.append(P_predicts)
        
        x_t_l.pop(0)
        P_t_l.pop(0)
        
        return x_t_l, P_t_l

    def batch_filter(self, x_t_l, P_t_l, zts, Rts):
        x_est_t_l = []
        P_est_t_l = []

        for idt in range(self.batch_Tg):
        
            x_predicts = x_t_l[idt]
            P_predicts = P_t_l[idt]
            zs = zts[idt]
            Rs = Rts[idt]
            
            for idx_s in range(len(x_predicts)):

                if self.cost[idx_s] is not None and self.cost[idx_s] < 0.01:
                    continue
                else:

                    x = x_predicts[idx_s]
                    P = P_predicts[idx_s]
                    z = zs[idx_s]
                    R = Rs[idx_s]

                    if z is not None:
                        x1, P1 = state_update(x, P, z, R)
                        x_predicts[idx_s] = x1
                        P_predicts[idx_s] = P1

            x_est_t_l.append(x_predicts)
            P_est_t_l.append(P_predicts)

        return x_est_t_l, P_est_t_l
    
    def calculate_cost(self, x_est_t_l):
        # print("Calculate cost!")
        
        if self.em_iteration_times == 0:
            self.em_iteration_times += 1
            return True

        cost_flag = False
        cost_t_l = []
        for idt in range(self.batch_Tg):
            t_id = self.t_buff[idt]
            
            cost_l = []
            for idx in range(len(self.target_state[t_id])):
                xn_1 = self.target_state[t_id][idx]
                xn = x_est_t_l[idt][idx]
                cost = (xn-xn_1).T @ inv(self.Q) @ (xn-xn_1)
                cost_l.append(cost)
            
            cost_t_l.append(cost_l)
        
        for idx in range(len(self.target_state[t_id])):

            cost_l = []
            for idt in range(self.batch_Tg):
                cost_l.append(cost_t_l[idt][idx])
            
            self.cost[idx] = np.sum(cost_l)/self.batch_Tg
            print(f"id: {idx} cost: {self.cost[idx]}")
            if self.cost[idx] >=0.01 and cost_flag==False:
                cost_flag = True
        
        return cost_flag


    def calculate_measures_and_covariance(self, w_tnsr):
        # print("synthetic measurements")

        zts = [None]*len(w_tnsr)
        Rts = [None]*len(w_tnsr)
        for idt in range(self.batch_Tg):
            w_nsr = w_tnsr[idt]
            measurements = self.meas_buff[idt]
            zs = [None]*len(w_nsr)
            Rs = [None]*len(w_nsr)

            for idx_s, wns in enumerate(w_nsr):

                temp_z = np.zeros((2, 1), dtype=np.float)
                for idx_r, wnsr in enumerate(wns):
                    temp_z += wnsr*measurements[idx_r]
                
                sum_wns = np.sum(wns)

                if sum_wns != 0:
                    temp_z = temp_z/sum_wns
                    temp_R = self.R/sum_wns

                    zs[idx_s] = temp_z
                    Rs[idx_s] = temp_R
            
            zts[idt] = zs
            Rts[idt] = Rs
        
        return zts, Rts

    def calculate_weight_s(self, x_t_l):
        
        w_tnsr = []
        for idx in range(self.batch_Tg):
            measurements = self.meas_buff[idx]
            x_predicts = x_t_l[idx]
            
            t_id = self.t_buff[idx]
            pi_s = self.get_prior_prob(measurements.shape[0], len(x_predicts))

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
            
            w_tnsr.append(w_nsr)

        return w_tnsr
    
    def update_info(self, x_est_l, P_est_l):
        
        for idt in range(self.batch_Tg):
            t_id = self.t_buff[idt]
        
            self.target_state[t_id] = np.stack(x_est_l[idt], axis=0)
            self.P[t_id] = np.stack(P_est_l[idt], axis=0)

    
    def get_prior_prob(self, z_size, x_size):
        # print("Compute prior probabilities!")

        meas_size = z_size
        target_size = x_size
        expect_mu = self.noise_expected

        pd = compute_detection_prob(meas_size, target_size)
        p_meas_size = compute_poisson_prob(expect_mu, meas_size)
        p_meas_tag = compute_poisson_prob(expect_mu, np.max((meas_size-target_size, 0)))
        
        pi_s = pd/meas_size*p_meas_tag/(pd*p_meas_tag + (1-pd)*p_meas_size)


        return pi_s

        
        

            
            
