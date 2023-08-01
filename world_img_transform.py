import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import scipy 

from matplotlib.widgets import RectangleSelector

class world_img_transform:

    # .mat and .json file path
    setting_path = None
    voltage_cali = None
    video_path = None

    # Chessboard inner corner number
    pattern_size = None

    # Template matching info
    interval = 3
    threshold = 0.7
    template_width = 40
    x1, x2, y1, y2 = None
    video = None

    # Time info
    wall_board_time = None
    spad_board_time = None
    begin_time = None
    end_time = None

    # Intrinsic params
    camera_matrix = None

    # Extrinsic params since scene is fixed
    R, t = None

    def __init__(self, 
                 setting_path,
                 interval=3,
                 threshold=0.7,
                 template_width=40):
        
        self.setting_path = setting_path
        f = open(setting_path, 'r')
        content = json.load(f)
        
        self.voltage_cali = scipy.io.loadmat(content['input_params_path'])
        self.video_path = content['video_path']
        
        self.wall_board_time = content['wall_board_time']
        self.spad_board_time = content['spad_board_time']
        self.begin_time = content['begin_time']
        self.end_time = content['end_time']
        
        self.interval = interval
        self.threshold = threshold
        self.template_width = template_width
    

    def line_select_callback(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.x1, self.x2, self.y1, self.y2 = y1, y2, x1, x2
        print("Center: ", (self.x1 + self.x2) // 2, ",", (self.y1 + self.y2) // 2)


    # get the template frame
    def get_template_frame(self, time, x_1, x_2, y_1, y_2):
        if not self.video.isOpened():
            print("Could not open video file")  
        self.video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        ret, frame = self.video.read()
        if ret:
            print("Frame read successfully")
        template = frame[x_1:x_2, y_1:y_2]
        return template

    # get the point in each frame
    def get_points(video, begin_time, end_time, interval, threshold, template):
        points = []
        indicator = interval - 1
        video.set(cv2.CAP_PROP_POS_MSEC, begin_time)  
        while True:
            if video.get(cv2.CAP_PROP_POS_MSEC) > end_time:
                break
            ret, frame = video.read()
            if not ret:
                break
            indicator += 1
            indicator %= interval
            if indicator != 0:
                continue
            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            center_point = []
            for pt in zip(*loc[::-1]):
                center_point.append(pt)
            if not len(center_point) == 0:
                point = (np.mean([x[0] for x in center_point]), np.mean([x[1] for x in center_point]))
                points.append(point)
                print("dot detected at time: ", video.get(cv2.CAP_PROP_POS_MSEC) / 1000, "Pos:", point)
        return points


    # cluster the points
    def cluster_points(points, threshold=8):
        point_center = []
        output_point = []
        for point in points:
            if point_center == []:
                point_center.append(point)
                continue
            # if the distance between the point and the center is less than 3, then add the point to the center
            if np.sqrt((point[0] - point_center[0][0]) ** 2 + (point[1] - point_center[0][1]) ** 2) < threshold:
                point_center.append(point)
            else:
                if len(point_center) == 1:
                    point_center = [point]
                    continue
                # add the mean of the points in the center to the output and the mean should be int
                output_point.append([np.mean([x[0] for x in point_center]), np.mean([x[1] for x in point_center])])
                point_center = [point]
        if point_center != []:
            output_point.append([np.mean([x[0] for x in point_center]), np.mean([x[1] for x in point_center])])
        return output_point

    def find_center(video, template_time, content):
        video.set(cv2.CAP_PROP_POS_MSEC, template_time * 1000)
        ret, frame = video.read()
        if not ret:
            print("Frame read unsuccessfully")
            exit()
        fig, ax = plt.subplots()
        ax.imshow(frame)
        temp = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=5, minspany=5,
            spancoords='pixels',
            rectprops=dict(facecolor='red', edgecolor = 'black', alpha=0.1, fill=True),
            interactive=True)
        plt.show()
        x = (glob_x1 + glob_x2) // 2
        y = (glob_y1 + glob_y2) // 2
        return x, y
        