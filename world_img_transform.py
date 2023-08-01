import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import scipy 

class world_img_transform:

    # .mat and .json file path
    detection_path = None
    voltage_path = None
    setting_path = None
    output_path = None
    
    # chessboard inner corner number
    pattern_size = None

    def __init__(self, setting_path,):
        
        self.setting_path = setting_path
        f = open(setting_path, 'r')
        content = json.load(f)
        self.voltage = content['voltage_path']
        self.wall_board_time = content['wall_board_time']
        self.spad_board_time = content['spad_board_time']
        self.video_path = content['video_path']
        
