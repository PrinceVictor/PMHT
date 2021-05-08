import numpy as np

from utils.motion_model import cv_state_trans_matrix

def measurement_matrix():
    return np.mat([[1, 0, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float)

def get_process_noise_matrix(delta_t, sigma=0.01):
    Q = np.power(sigma, 2) * np.mat([[np.power(delta_t, 4)/4, np.power(delta_t, 3)/2, 0., 0.],
                                     [np.power(delta_t, 3)/2, np.power(delta_t, 2), 0., 0.  ],
                                     [0., 0., np.power(delta_t, 4)/4, np.power(delta_t, 3)/2],
                                     [0., 0., np.power(delta_t, 3)/2, np.power(delta_t, 2)  ]], dtype=np.float)
    
    return Q

def get_measurement_noise_matrix(sigma=100):
    R = np.power(sigma, 2) * np.mat(np.eye(2, dtype=np.float))
    
    return R

def state_predict(x, P, Q, delta_t):
    
    cv_state_trans_mat = cv_state_trans_matrix(delta_t, 2, 2)
    x_predict = cv_state_trans_mat * np.mat(x)
    P_predict = cv_state_trans_mat * np.mat(P) * cv_state_trans_mat.T + Q

    return x_predict, P_predict

def state_update(x_predict, P_predict, z, R):
    
    H = measurement_matrix()
    k_gain = P_predict * H.T * (H * P_predict * H.T + R).I
    x_estimate = x_predict + k_gain * (z - H*x_predict)
    P_estimate = P_predict - k_gain*H*P_predict
    
    return x_estimate, P_estimate