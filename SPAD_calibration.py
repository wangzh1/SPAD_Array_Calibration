import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import scipy 
import scipy.io
import torch

from torch.utils.data import Dataset
from matplotlib.widgets import RectangleSelector


class Model(torch.nn.Module):

    """
    MLP for voltage-coordinates mapping.
    One hidden layer to avoid overfitting.
    """

    def __init__(self,num_i,num_h,num_o):
        super(Model, self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.tanh1=torch.nn.Tanh()
        self.linear2=torch.nn.Linear(num_h,num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)

        return x


class MyData(Dataset):
    
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    def __len__(self):
        return self.input.shape[0]
        


class SPAD_calibration:

    # Path info
    setting_path = None
    video_path = None
    detected_pixels_path = None

    # Chessboard inner corner number
    pattern_size = None

    # Input params
    voltage_cali = None

    # Template matching info
    interval = 3
    threshold = 0.7
    template_width = 40
    video = None
    pattern_size = (15, 15)

    # Time info
    wall_board_time = None
    spad_board_time = None
    template_time = None
    begin_time = None
    end_time = None

    # Intrinsic params
    camera_matrix = None
    dist_coeffs = None
    rvecs = None
    tvecs = None

    def __init__(self, 
                 setting_name,
                 interval=3,
                 threshold=0.7,
                 template_width=40,
                 pattern_size=(15,15)):
        
        setting_path = 'setting/' + setting_name + '.json'
        f = open(setting_path, 'r')
        content = json.load(f)
        
        # Path info
        self.video_path = 'video/' + setting_name + '.MP4'
        self.detected_pixels_path = 'detected_pixels/' + setting_name + '.mat'

        # Input params
        input_params_path = 'input_params/' + setting_name + '.mat'
        self.voltage_cali = scipy.io.loadmat(input_params_path) # (2, num_of_points)

        # Time info
        self.wall_board_time = content['wall_board_time']
        self.spad_board_time = content['spad_board_time']
        self.template_time = content['template_time']
        self.begin_time = content['begin_time']
        self.end_time = content['end_time']
        
        # read the video
        self.video = cv2.VideoCapture(self.video_path)

        # Template matching info
        self.interval = interval
        self.threshold = threshold
        self.template_width = template_width
        self.pattern_size = pattern_size
    
  # ----------- template matching -> pixels -----------
    def find_center(self):
        """
        Manually select the template for further matching.
        """

        # Define call back function for RectSelector
        def line_select_callback(eclick, erelease):
            global x1, x2, y1, y2
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            print("Center: ", (x1 + x2) // 2, ",", (y1 + y2) // 2)

        # Read frame from the given video
        self.video.set(cv2.CAP_PROP_POS_MSEC, self.template_time * 1000)
        ret, frame = self.video.read()

        if not ret:
            print("Frame read unsuccessfully")
            exit()
        
        _, ax = plt.subplots()
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
        
        return (x1 + x2) // 2, (y1 + y2) // 2
        

    # get the frame from the video at the given time
    def get_video_frame(self, time):
        
        time *= 1000
        self.video.set(cv2.CAP_PROP_POS_MSEC, time)
        success, image = self.video.read()
        if success:
            return image
        
        return None


    # get the template frame
    def get_template_frame(self, x1, x2, y1, y2):
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
            print(x1, x2, y1, y2)
        
        return frame[y1:y2, x1:x2]


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


    def cluster_points(self, points):
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
            if np.sqrt((point[0] - point_center[0][0]) ** 2 + (point[1] - point_center[0][1]) ** 2) < self.threshold:
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

    
    def template_matching(self):
        
        # get template frame
        center_x, center_y = self.find_center()
        x1, x2, y1, y2 = center_x - self.template_width // 2, \
                         center_x + self.template_width // 2, \
                         center_y - self.template_width // 2, \
                         center_y + self.template_width // 2
        template = self.get_template_frame(x1, x2, y1, y2)

        # show the template frame
        plt.imshow(template)
        plt.show()
        
        # get the points
        points = self.get_points(template)
        
        # cluster the points
        detected_pixels = self.cluster_points(points)

        plt.figure()
        for dot_pos in detected_pixels:
            plt.scatter(dot_pos[0], 1080 - dot_pos[1], c='r', s=10)
        print(len(detected_pixels))
        mdic = {'detected_pixels':detected_pixels}
        
        scipy.io.savemat(self.detected_pixels_path, mdic)
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        plt.show()


  # ----------- scene cali -> points coords & spad coords -----------
    def calibrate_scene(self, time):
        """
        Detect chessboard corners in the frame at input time, return 
        """
        
        time *= 1000
        self.video.set(cv2.CAP_PROP_POS_MSEC, time) # gray scale image
        success, img = self.video.read()
        
        if not success:
            print("Frame read unsucceeded.")
            return

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img
        pixels = []
        coords = []

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (15, 15), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            pixels.append(corners)
            pixels = np.array(pixels)
            pixels_squeezed = np.squeeze(pixels)
        
        coord = np.zeros((15*15, 3), np.float32)

        # To make positive x-axis pointing at right and positive y-axis pointing at top,
        # so that the directions match the directions of voltages' change.
        coord[:, :2] = np.flip(np.flip(np.mgrid[0:15, 0:15].T.reshape(-1, 2), axis=0), axis=1) / 100 * 1.6
        coords.append(coord)

        _, camera_matrix, _, rvecs, tvecs = cv2.calibrateCamera(coords, pixels, gray.shape[::-1], None, None)

        return camera_matrix, rvecs, tvecs


    @staticmethod
    def img_to_world(img_pixels, rvecs, tvecs, camera_matrix):
        """
        img_pixels: (num_of_points, 2)
        rvecs, tvecs: tuple
        world_coords: (2, num_of_points)
        """

        img_pixels = img_pixels.T # (2, num_of_points) 
        img_pixels = np.concatenate((img_pixels, np.ones([1, img_pixels.shape[1]])), axis=0) # (3, num_of_points) 
        
        # Transfer rotation vector to rotation matrix, and get translation matrix from tuple.
        R, _ = cv2.Rodrigues(rvecs[0])
        t = tvecs[0]

        scale = (np.linalg.inv(R) @ t)[2, :] / \
                (np.linalg.inv(R) @ np.linalg.inv(camera_matrix) @ img_pixels)[2, :] # (1, num_of_points)
        
        world_coords = scale * (np.linalg.inv(R) @ np.linalg.inv(camera_matrix) @ img_pixels) - np.linalg.inv(R) @ t
        
        return world_coords[:2, :] # (2, num_of_points), z = 0



def initilization(coords, voltage):
    dtheta = 0.069719062113912
    center_index = np.where((np.abs(voltage[:,0]) < 1e-4) * (np.abs(voltage[:,1]) < 1e-4))[0]
    center_coords = coords[center_index,0:3]
    base_index = np.where((np.abs(voltage[:,0]) < 1e-4) + (np.abs(voltage[:,1]) < 1e-4))[0]
    base_coords = coords[base_index,0:3]
    base_voltage = voltage[base_index,0:2]
    G_point = np.array([center_coords[0,0], center_coords[0,1], 1]) + np.random.randn(3) * 0.01
    max_step = 1001
    for p in range(1001):
        num_search = 11
        GP_se = G_point.reshape(1,3) + np.random.randn(num_search,3) * 0.1
        GP_se = np.concatenate((GP_se, G_point.reshape(1,3)), axis = 0)
        GP_se = GP_se.reshape(1,num_search+1,3)

        l0 = np.sqrt(np.sum((GP_se - center_coords.reshape(1,1,3)) ** 2, axis = 2))
        li = np.sqrt(np.sum((GP_se - base_coords.reshape(-1,1,3)) ** 2, axis = 2))
        d = np.sqrt(np.sum((center_coords.reshape(1,1,3) - base_coords.reshape(-1,1,3)) ** 2, axis = 2))
        costheta = (l0 ** 2 + li ** 2 - d ** 2) / 2 / l0 / li
        theta_diff = np.arccos(costheta) - dtheta * np.sum(np.abs(base_voltage), axis = 1)[:,None]
        loss = np.sqrt(np.mean(theta_diff ** 2, axis = 0))

        index = np.where((np.abs(loss - np.nanmin(loss)) < 1e-8))
        index_i = index[0][0].item()
        if p % 500 == 0:
            pass
            print('Initialization:', p,'/',max_step-1,' Angle RMSE:', loss[index_i])
        
        G_point = GP_se[0,index_i,:]
    return G_point
    