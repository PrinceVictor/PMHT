import numpy as np


class MOT:
    def __init__(self, times, keep_T=3):
        print("Construct MOT")
        self.targets = [None] * times
        self.track_ids = [None] * times
        self.meas_buff = []
    
    def get_measurements(self, t_id, data):
        self.meas_buff.append(data)
    
    def data_association(self):