'''
Calibration part2: get the world position of each dot
calibrate the camera by the chessboard and get the transformation matrix
save the transformed points in a mat file with its corresponding voltage
'''
import cv2
import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# parameters
json_path = 'setting/C0016.json'
pattern_size = (15, 15)
square_size = 1
output_path = 'coords_voltages/coords_voltages_C0016.mat'
detection_path = 'detection_result/detection_result_C0016.mat'

f = open(json_path, 'r')
content = json.load(f)
voltage_path = content['voltage_path']
chessboard_time = content['chessboard_time']
video_path = content['video_path']


pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size


# get the frame from the video at the given time
def get_video_frame(video, time):
    time *= 1000
    video.set(cv2.CAP_PROP_POS_MSEC, time)
    success, image = video.read()
    if success:
        return image
    return None

def find_chess_board_corners(video, pattern_size, time):
    board_image = get_video_frame(video, time)
    if board_image is None:
        print("Image not found at time: ", time)
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([0,-1,0,-1,5,-1,0,-1,0])
    dst = cv2.filter2D(gray,-1,kernel)
    dst = ((dst >= 65) * 255 + (dst < 65) * 0).astype(np.uint8)

    found, corners = cv2.findChessboardCorners(dst, pattern_size)
    if not found:
        print("Corners not found at time: ", time)
        return None
    return corners, gray  

def calibrate_camera(pattern_points, corners, gray):
    world_points = []
    image_points = []
    world_points.append(pattern_points)
    image_points.append(corners)

    _, camera_matrix, _, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, gray.shape[::-1], None, None)

    R, _ = cv2.Rodrigues(rvecs[0])
    camera_position = -np.matrix(R).T * np.matrix(tvecs[0])
    camera_direction = np.matrix(R).T * np.matrix([0, 0, 1]).T
    return camera_matrix, tvecs, R

def img_to_world(img_point, R, tvecs, camera_matrix):
    tmp1 = np.dot(np.linalg.inv(R), np.dot(np.linalg.inv(camera_matrix), img_point.T))
    tmp2 = np.dot(np.linalg.inv(R), np.array(tvecs[0]).reshape(3).T)
    s = tmp2[2] / tmp1[2]
    world_img_point = np.dot(np.linalg.inv(R), (np.dot(np.linalg.inv(camera_matrix), img_point.T) * s - np.array(tvecs[0]).reshape(3)).T)
    return world_img_point

def get_world_pos(points):
    dim_1 = []
    dim_2 = []
    world_img_points = []
    for point in points:
        world_img_points.append(img_to_world(np.array([point[0], point[1], 1]), R, tvecs, camera_matrix))
    for i in range(len(world_img_points)):
        dim_1.append(world_img_points[i][0])
        dim_2.append(world_img_points[i][1])
    dim_1 = np.array(dim_1) 
    dim_2 = np.array(dim_2)
    return dim_1, dim_2


if __name__ == "__main__":
    video = cv2.VideoCapture(video_path)
    # find the corners
    corners, gray = find_chess_board_corners(video, pattern_size, chessboard_time)
    # calibrate the camera
    camera_matrix, tvecs, R = calibrate_camera(pattern_points, corners, gray)
    # save the data (A, b)
    out_pt = sio.loadmat(detection_path)
    output_point = out_pt['pt']
    dim_1, dim_2 = get_world_pos(output_point)
    A = np.column_stack((dim_1, dim_2, np.zeros_like(dim_1), np.ones_like(dim_1)))  # world position
    voltage = sio.loadmat(voltage_path)
    v_x = voltage['voltage_x'][0]
    v_y = voltage['voltage_y'][0]
    b = np.column_stack((v_x, v_y))
    mdic = {"coords": A, "voltages": b}
    sio.savemat(output_path, mdic)

    # show the points in the world coordinate
    plt.figure()
    for i in range(len(dim_1)):    
        plt.scatter(dim_1[i], dim_2[i], c='r', s=10)
    plt.scatter(dim_1[0], dim_2[0], c='g', s=30)
    for i in range(15):
        for j in range(15):
            plt.scatter(i, j, c='b', s=10)
    plt.scatter(0, 0, c='y', s=30)
    plt.scatter(14, 0, c='r', s=30) # x
    plt.scatter(0, 14, c='g', s=30) # y

    plt.show()