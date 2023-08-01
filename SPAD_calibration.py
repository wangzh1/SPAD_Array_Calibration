import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import scipy 

from matplotlib.widgets import RectangleSelector

class SPAD_calibration:

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
    template_time = None
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
        self.template_time = content['template_time']
        self.begin_time = content['begin_time']
        self.end_time = content['end_time']
        
        self.interval = interval
        self.threshold = threshold
        self.template_width = template_width
    

    def find_center(self):
        """
        Manually select the template for further matching.
        """

        # Define call back function for RectSelector
        def line_select_callback(eclick, erelease):
            self.x1, self.y1 = int(eclick.xdata), int(eclick.ydata)
            self.x2, self.y2 = int(erelease.xdata), int(erelease.ydata)
            print("Center: ", (self.x1 + self.x2) // 2, ",", (self.y1 + self.y2) // 2)

        # Read frame from the given video
        self.video.set(cv2.CAP_PROP_POS_MSEC, self.template_time * 1000)
        ret, frame = self.video.read()

        if not ret:
            print("Frame read unsuccessfully")
            exit()
        
        fig, ax = plt.subplots()
        ax.imshow(frame)

        temp = RectangleSelector(
            ax, 
            line_select_callback,
            drawtype='box', useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=5, minspany=5,
            spancoords='pixels',
            rectprops=dict(facecolor='red', edgecolor = 'black', alpha=0.1, fill=True),
            interactive=True)
        
        plt.show()
        
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2
        

    # get the template frame
    def get_template_frame(self):
        """
        Return the template image.
        """
        
        if not self.video.isOpened():
            print("Could not open video file")  
        self.video.set(cv2.CAP_PROP_POS_MSEC, self.template_time * 1000)
        ret, frame = self.video.read()
       
        # Succefully read a frame
        if ret:
            print("Frame read successfully")
        template = frame[self.x1:self.x2, self.y1:self.y2]
        
        return template


    # get the point in each frame
    def get_points(self, template):
        """
        Get all points related to the selected template.
        """
        
        points = []
        indicator = self.interval - 1
        self.video.set(cv2.CAP_PROP_POS_MSEC, self.begin_time * 1000)  
        while True:
            if self.video.get(cv2.CAP_PROP_POS_MSEC) > self.end_time * 1000:
                break
            ret, frame = self.video.read()
            if not ret:
                break
            indicator += 1
            indicator %= self.interval
            if indicator != 0:
                continue
            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.threshold)
            center_point = []
            for pt in zip(*loc[::-1]):
                center_point.append(pt)
            if not len(center_point) == 0:
                point = (np.mean([x[0] for x in center_point]), np.mean([x[1] for x in center_point]))
                points.append(point)
                print("dot detected at time: ", self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000, "Pos:", point)
        
        return points


    def cluster_points(self, points, threshold=8):
        """
        Cluster all pixels related to the template selected before.
        """

        point_center = []
        clustered_points = []

        for point in points:
            if point_center == []:
                point_center.append(point)
                continue
            
            # If the distance between the point and the center is less than threshold, then add the point to the center
            if np.sqrt((point[0] - point_center[0][0]) ** 2 + (point[1] - point_center[0][1]) ** 2) < threshold:
                point_center.append(point)
            else:
                if len(point_center) == 1:
                    point_center = [point]
                    continue

                # Add the mean of the points in the center to the output and the mean should be int
                clustered_points.append([np.mean([x[0] for x in point_center]), np.mean([x[1] for x in point_center])])
                point_center = [point]

        if point_center != []:
            clustered_points.append([np.mean([x[0] for x in point_center]), np.mean([x[1] for x in point_center])])
        
        return clustered_points

    