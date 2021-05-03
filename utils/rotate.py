import numpy as np

rotate_dict = {"yaw": 0, "pitch": 1, "roll": 2}

def rotate_matrix(index, radian):
    def rotate_yaw(yaw):
        return np.mat([[np.cos(yaw), np.sin(yaw), 0],
                       [-np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]], dtype=np.float)

    def rotate_pitch(pitch):
        return np.mat([[np.cos(pitch), 0, -np.sin(pitch)],
                       [0, 1, 0],
                       [np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float)

    def rotate_roll(roll):
        return np.mat([[1, 0, 0],
                       [0, np.cos(roll), np.sin(roll)],
                       [0, -np.sin(roll), np.cos(roll)]], dtype=np.float)

    rotate_mat = [rotate_yaw, rotate_pitch, rotate_roll]

    return rotate_mat[index](radian)


def direction_cosine_matrix3x3(*args, **kwargs):

    rotate_mat = np.mat(np.identity(3, dtype=np.float))
    if len(args) > 0:
        for index, radian in enumerate(args):
            rotate_mat = rotate_matrix(index, radian)*rotate_mat
    elif len(kwargs) > 0:
        for index, name in enumerate(kwargs):
            rotate_mat = rotate_matrix(rotate_dict[name], kwargs[name])*rotate_mat

    return rotate_mat




