import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import *

class Target:
    def __init__(self, id, delta_t):
        self.id = id
        self.delta_t = delta_t
        self.state = np.zeros((4, 1), dtype=np.float)
        self.P = np.zeros((4, 4), dtype=np.float)
        self.Q = np.round(get_process_noise_matrix(self.delta_t, sigma=0.85))
        
        self.keep_times = 0
        self.tracked = 0
        self.candidate = 1
        self.vanish = 0
    
    def update(self):
        self.keep_times += 1

class MOT:
    def __init__(self, times, delta_t, keep_T=3):
        print("Construct MOT")
        self.targets = [[]] * times
        self.meas_buff = []
        self.keep_T = keep_T
        self.target_id_seed = 0
        self.delta_t = delta_t
        self.cost_threshold = 1000
    
    def create_target_id(self):
        
        self.target_id_seed+=1
        return self.target_id_seed-1
    
    def run_track(self, t_id, meas):
        print(f"Running Track with Time ID:{t_id}")
        
        self.get_measurements(meas)

        if t_id == 0:
            self.track_init()
        else:
            self.targets_predict(t_id)
            self.data_association(t_id)
    
    def get_measurements(self, data):
        self.meas_buff.append(data)
    
    def targets_predict(self, t_id):
        
        target_nums = len(self.targets[t_id-1])

        self.targets[t_id] = self.targets[t_id-1]   
        for x_id in range(target_nums):
            x = self.targets[t_id][x_id].state
            P = self.targets[t_id][x_id].P
            Q = self.targets[t_id][x_id].Q

            x_pre, P_pre = state_predict(x, P, Q, self.delta_t)
            self.targets[t_id][x_id].state = x_pre
            self.targets[t_id][x_id].P = P_pre

 

    def track_init(self):
        print("Track initialization!!!")

        xs = np.zeros((self.meas_buff[0].shape[0], 4, 1), dtype=np.float)
        
        for x_id, meas in enumerate(self.meas_buff[0]):
            target = Target(id=self.create_target_id(), delta_t=self.delta_t)
            target.state[0] = meas[0]
            target.state[2] = meas[1]
            self.targets[0].append(target)
            

    def calculate_cost(self, x, y):
        x_dist = x[0][0] - y[0][0]
        y_dist = x[2][0] - y[1][0]
        
        cost = np.sqrt(x_dist**2 + y_dist**2)
        return cost if cost <= self.cost_threshold else self.cost_threshold

    def data_association(self, t_id):
        print(f"Data Association!")
        
        target_num = len(self.targets[t_id])
        meas_num = len(self.meas_buff[t_id])

        cost_mat = np.zeros(shape=(target_num, meas_num), dtype=np.float)
        for x_id in range(target_num):
            for y_id in range(meas_num):
                cost = self.calculate_cost(self.targets[t_id][x_id].state, 
                                           self.meas_buff[t_id][y_id])
                cost_mat[x_id][y_id] = cost

        assignment = [-1] * target_num
        row_id, col_id = linear_sum_assignment(cost_mat)
        
        np.set_printoptions(threshold=np.inf)
        print(target_num, meas_num)
        print(cost_mat)
        print(row_id)
        print(col_id)

        for i in range(len(row_id)):
            assignment[row_id[i]] = col_id[i]
        
        raise SystemExit

