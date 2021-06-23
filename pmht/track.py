import numpy as np

class Target:
    def __init__(self)

class MOT:
    def __init__(self, times, keep_T=3):
        print("Construct MOT")
        self.targets = [None] * times
        self.track_ids = [None] * times
        self.meas_buff = []
        self.keep_T = keep_T
    
    def run_track(self, t_id, meas):
        self.get_measurements(meas)

        if t_id == 0:
            self.track_init()
        else:
            self.data_association(t_id)
    
    def get_measurements(self, data):
        self.meas_buff.append(data)
    
    def track_init(self):
        print(self.meas_buff[0].shape)

        xs = np.zeros((self.meas_buff[0].shape[0], 4, 1), dtype=np.float)

        for x_id, x in enumerate(self.meas_buff[0]):
            print(x_id)
            xs[x_id, ]
            # print(self.meas_buff[0][x_id])


    def data_association(self, t_id):
        print(f"Data Association!")
        # if t_id == 0:
