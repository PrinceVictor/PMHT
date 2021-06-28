import numpy as np
from scipy.optimize import linear_sum_assignment

import copy

import pmht.kalman as kalman

class Target:
    def __init__(self, id, delta_t):
        self.id = id
        self.delta_t = delta_t
        self.state = np.zeros((4, 1), dtype=np.float)
        self.P = np.zeros((4, 4), dtype=np.float)
        self.Q = np.round(kalman.get_process_noise_matrix(self.delta_t, sigma=0.85))
        
        self.keep_times = 0
        self.unmatched_times = 0
        self.occur_times = 0
        self.tracked = 0
        self.candidate = 1
        self.vanish = 0
    
    def state_predict(self):

        x_pre, P_pre = kalman.state_predict(self.state, 
                                            self.P, 
                                            self.Q, 
                                            self.delta_t)
        self.state = x_pre
        self.P = P_pre

    def state_update(self, meas=None, R=None):
        self.status_update(meas)

        if meas is not None:
            x_est, P_est = kalman.state_update(self.state, self.P,
                                               meas, R)
            self.state = x_est
            self.P = P_est
            

    def status_update(self, meas_flag):
        self.occur_times += 1

        if meas_flag is None:
            self.unmatched_times += 1
        else:
            if self.candidate == 1:
                self.keep_times += 1
            elif self.tracked == 1:
                self.unmatched_times = 0
                self.keep_times = 0
                self.occur_times = 0
        
        if self.unmatched_times >= 3:
            self.vanish = 1
            self.tracked = 0
            self.candidate = 0
        elif self.keep_times>=3 and self.occur_times<=5:
            self.tracked = 1
            self.keep_times = 0
            self.occur_times = 0
            self.unmatched_times = 0
            self.candidate = 0

class MOT:
    def __init__(self, times, delta_t, keep_T=3, meas_sigma=10):
        print("Construct MOT")
        self.targets = [[]] * times
        self.meas_buff = []
        self.keep_T = keep_T
        self.target_id_seed = 0
        self.delta_t = delta_t
        self.cost_threshold = 300

        self.R = kalman.get_measurement_noise_matrix(sigma=meas_sigma)
    
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
            assignment = self.data_association(t_id)
            self.targets_update(t_id, assignment)
            self.delete_targets(t_id)
            self.create_new_targets(t_id, assignment)
        
        print(f"total target num: {len(self.targets[t_id])}")
        # for x_id in range(3):
        #     target = self.targets[t_id][x_id]
        #     print(f"target id: {target.id}, target state:\n{target.state}")
    
    def get_measurements(self, data):
        self.meas_buff.append(data)
    
    def track_init(self):
        print("Track initialization!!!")

        xs = np.zeros((self.meas_buff[0].shape[0], 4, 1), dtype=np.float)     
        for x_id, meas in enumerate(self.meas_buff[0]):
            target = Target(id=self.create_target_id(), delta_t=self.delta_t)
            target.state[0] = meas[0]
            target.state[2] = meas[1]
            self.targets[0].append(target)
    
    def targets_predict(self, t_id):
        
        target_nums = len(self.targets[t_id-1])

        self.targets[t_id] = copy.deepcopy(self.targets[t_id-1]) 
        for x_id in range(target_nums):
            self.targets[t_id][x_id].state_predict()
            

    def calculate_cost(self, x, y):
        x_dist = x[0][0] - y[0][0]
        y_dist = x[2][0] - y[1][0]
        
        cost = np.sqrt(x_dist**2 + y_dist**2)
        return cost if cost <= self.cost_threshold else self.cost_threshold*10

    def data_association(self, t_id):
        print(f"Data Association!")
        
        target_num = len(self.targets[t_id])
        meas_num = len(self.meas_buff[t_id])

        cost_mat = np.zeros(shape=(target_num, meas_num), dtype=np.float)
        assignment = np.zeros(shape=(target_num), dtype=np.int)
        assignment.fill(-1)

        for x_id in range(target_num):
            for y_id in range(meas_num):
                cost = self.calculate_cost(self.targets[t_id][x_id].state, 
                                           self.meas_buff[t_id][y_id])
                cost_mat[x_id][y_id] = cost
        
        # np.set_printoptions(threshold=np.inf)
        # print(cost_mat[:3, :3])
        # print(self.meas_buff[t_id][:3])

        row_id, col_id = linear_sum_assignment(cost_mat)
        
        for i in range(len(row_id)):
            if cost_mat[row_id[i]][col_id[i]] < self.cost_threshold:
                assignment[row_id[i]] = col_id[i]
        
        return assignment
    
    def targets_update(self, t_id, assignment):
        print(f"Targets Update!")

        targets_num = len(self.targets[t_id])
        for x_id, target in enumerate(self.targets[t_id]):
            
            meas_id = assignment[x_id]
            if meas_id == -1:
                target.state_update()
            else: 
                target.state_update(self.meas_buff[t_id][meas_id], 
                                    self.R)
            
            # self.targets[t_id][x_id] = target
    
    def create_new_targets(self, t_id, assignment):
        print(f"Create New Targets!")

        for y_id, meas in enumerate(self.meas_buff[t_id]):
            if y_id not in assignment:
                target = Target(id=self.create_target_id(), delta_t=self.delta_t)
                target.state[0] = meas[0]
                target.state[2] = meas[1]
                self.targets[t_id].append(target)
        
    def delete_targets(self, t_id):
        print(f"Delete Targets!")

        targets_num = len(self.targets[t_id])
        print(f"before delete {targets_num}")

        tracked_count = 0
        erased_count = 0
        for x_id in reversed(range(targets_num)):

            if self.targets[t_id][x_id].vanish == 1:
                self.targets[t_id].pop(x_id)
                erased_count += 1
            elif self.targets[t_id][x_id].tracked == 1:
                tracked_count += 1
        
        print(f"after delete {len(self.targets[t_id])} erased {erased_count} real tracked {tracked_count}")
    
    def statistics(self):
        for t_id, targets in enumerate(self.targets):
            print(f"T:{t_id} targets num {len(targets)}")



