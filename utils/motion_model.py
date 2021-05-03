import numpy as np


def cv_state_trans_matrix(delta_t, dims=3):
    state_trans_mat = np.mat([[1., delta_t, 0.5*delta_t*delta_t, 0., 0., 0., 0., 0., 0.],
                              [0., 1., delta_t, 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., delta_t, 0.5*delta_t*delta_t, 0., 0., 0.],
                              [0., 0., 0., 0., 1., delta_t, 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 1., delta_t, 0.5*delta_t*delta_t],
                              [0., 0., 0., 0., 0., 0., 0., 1., delta_t],
                              [0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float)

    if dims == 3:
        return state_trans_mat
    elif dims == 2:
        return state_trans_mat[0:6, 0:6]
