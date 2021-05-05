import numpy as np

def cv_state_trans_matrix(delta_t, state_dims=2, space_dims=2):
    state_trans_mat = np.mat([[1., delta_t, 0.5*delta_t*delta_t, 0., 0., 0., 0., 0., 0.],
                              [0., 1., delta_t, 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., delta_t, 0.5*delta_t*delta_t, 0., 0., 0.],
                              [0., 0., 0., 0., 1., delta_t, 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 1., delta_t, 0.5*delta_t*delta_t],
                              [0., 0., 0., 0., 0., 0., 0., 1., delta_t],
                              [0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float)
    
    state_trans_mat = state_trans_mat[0:3*space_dims, 0:3*space_dims]
    
    if state_dims == 2:
        state_trans_mat = np.delete(state_trans_mat, np.arange(1, 3)*3-1, axis=0)
        state_trans_mat = np.delete(state_trans_mat, np.arange(1, 3)*3-1, axis=1)
    return state_trans_mat
