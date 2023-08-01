import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio

video_path = 'video/C0016.MP4'
output_path = 'SPAD_array_cali/array_data_C0016.mat'

pattern_size = (15, 15)
square_size = 1

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

def get_video_frame(video, time):
    time *= 1000
    video.set(cv2.CAP_PROP_POS_MSEC, time)
    success, frame = video.read()
    return frame if success else None

# def find_chess_board_corners(board_image, pattern_size):
#     gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
#     kernel = np.array([0,-1,0,-1,5,-1,0,-1,0])
#     dst = cv2.filter2D(gray,-1,kernel)
#     dst = ((dst >= 65) * 255 + (dst < 65) * 0).astype(np.uint8)

#     found, corners = cv2.findChessboardCorners(dst, pattern_size)
#     if not found:
#         print("Corners not found")
#         return None
#     return corners, gray

def find_chess_board_corners(video, pattern_size, time):
    board_image = get_video_frame(video, time)
    if board_image is None:
        print("Image not found at time: ", time)
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([0,-1,0,-1,5,-1,0,-1,0])
    dst = cv2.filter2D(gray,-1,kernel)
    dst = ((dst >= 65) * 255 + (dst < 65) * 0).astype(np.uint8)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)
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

def get_world_pos(points, R, tvecs, camera_matrix):
    dim_1 = []
    dim_2 = []
    dim_3 = []
    world_img_points = []
    for point in points:
        world_img_points.append(img_to_world(np.array([point[0], point[1], 1]), R, tvecs, camera_matrix))
    for i in range(len(world_img_points)):
        dim_1.append(world_img_points[i][0])
        dim_2.append(world_img_points[i][1])
        dim_3.append(world_img_points[i][2])
    dim_1 = np.array(dim_1) 
    dim_2 = np.array(dim_2)
    dim_3 = np.array(dim_3)
    return dim_1, dim_2

def flip_corners(corners, pattern_size):
    if corners[0][0] < corners[-1][0] and corners[0][1] < corners[-1][1]:
        corners = corners.reshape(pattern_size[0], pattern_size[1], 2)
        corners = np.rot90(corners, -1)
        corners = corners.reshape(pattern_size[0] * pattern_size[1], 2)
    elif corners[0][0] > corners[-1][0] and corners[0][1] > corners[-1][1]:
        corners = corners.reshape(pattern_size[0], pattern_size[1], 2)
        corners = np.rot90(corners, 1)
        corners = corners.reshape(pattern_size[0] * pattern_size[1], 2)
    elif corners[0][0] > corners[-1][0] and corners[0][1] < corners[-1][1]:
        corners = np.flip(corners, axis=0)
        corners = corners.reshape(pattern_size[0] * pattern_size[1], 2)
    if corners[0][0] < corners[-1][0] and corners[0][1] > corners[-1][1]:
        if abs(corners[0][0] - corners[1][0]) > abs(corners[0][1] - corners[1][1]):
            return corners
        corners = corners.reshape(pattern_size[0], pattern_size[1], 2)
        corners = corners.transpose((1, 0, 2))
        corners = corners.reshape(pattern_size[0] * pattern_size[1], 2)
        return corners
    print('An error has occurred and additional conditions need to be added')
    return None # other condition need to be added


if __name__ == '__main__':
    video = cv2.VideoCapture(video_path)

    # ARRAY
    print('Finding corners...')
    corners, gray = find_chess_board_corners(video, pattern_size, 236)

    corners = np.squeeze(corners)
    corners = flip_corners(corners, pattern_size)

    # sio.savemat('temp_array.mat', {'array': corners})

    corners = np.expand_dims(corners, axis=1)
    # calibrate the camera
    camera_matrix, tvecs, R = calibrate_camera(pattern_points, corners, gray)
    # get the world position of the corners
    corners = np.squeeze(corners)
    print(corners.shape)
    dim1, dim2 = get_world_pos(corners, R, tvecs, camera_matrix)
    dim3 = np.zeros(len(dim1))
    # camera position
    camera_position = -np.matrix(R).T * np.matrix(tvecs[0])
    camera_direction = np.matrix(R).T * np.matrix([0, 0, 1]).T
    camera_up_direction = np.matrix(R).T * np.matrix([0, 1, 0]).T
    camera_left_direction = np.matrix(R).T * np.matrix([1, 0, 0]).T

    corners = np.concatenate((dim1[:, None], dim2[:, None], dim3[:, None]), axis=1)
    corners -= camera_position.T
    camera_position = np.array([0, 0, 0])
    corners = R @ corners.T
    camera_direction = R @ camera_direction
    camera_up_direction = R @ camera_up_direction
    camera_left_direction = R @ camera_left_direction
    dim1, dim2, dim3 = corners[0], corners[1], corners[2]
    temp1, temp2, temp3 = dim1, dim2, dim3

    # plot the corners and camera position and direction in 3d
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(dim1, dim2, dim3, c='r', marker='o')
    # ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='b', marker='o')
    # ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
    #           camera_direction[0], camera_direction[1], camera_direction[2], length=20, normalize=True)
    # ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
    #           camera_up_direction[0], camera_up_direction[1], camera_up_direction[2], length=5, normalize=True, color='r')
    # ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
    #           camera_left_direction[0], camera_left_direction[1], camera_left_direction[2], length=5, normalize=True, color='g')
    # ax.scatter(dim1[0], dim2[0], dim3[0], c='g', marker='o', s=100)
    # ax.scatter(dim1[1], dim2[1], dim3[1], c='g', marker='o', s=100)
    # plt.show()

    # WALL
    print('Finding corners...')
    corners, gray = find_chess_board_corners(video, pattern_size, 229)
    corners = np.squeeze(corners)
    corners = flip_corners(corners, pattern_size)

    # sio.savemat('temp_wall.mat', {'wall': corners})

    corners = np.expand_dims(corners, axis=1)
    if corners is None:
        print('corners not found')
        exit()
    print('Found')
    # calibrate the camera
    camera_matrix, tvecs, R = calibrate_camera(pattern_points, corners, gray)
    # get the world position of the corners
    corners = np.squeeze(corners)
    dim1, dim2 = get_world_pos(corners, R, tvecs, camera_matrix)
    dim3 = np.zeros(len(dim1))
    # camera position
    camera_position = -np.matrix(R).T * np.matrix(tvecs[0])
    camera_direction = np.matrix(R).T * np.matrix([0, 0, 1]).T
    camera_up_direction = np.matrix(R).T * np.matrix([0, -1, 0]).T
    camera_left_direction = np.matrix(R).T * np.matrix([-1, 0, 0]).T
    
    corners = np.concatenate((dim1[:, None], dim2[:, None], dim3[:, None]), axis=1)
    corners -= camera_position.T
    camera_position = np.array([0, 0, 0])
    corners = R @ corners.T
    camera_direction = R @ camera_direction
    camera_up_direction = R @ camera_up_direction
    camera_left_direction = R @ camera_left_direction
    dim1, dim2, dim3 = corners[0], corners[1], corners[2]
    
    # plot the corners and camera position and direction in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dim1, dim2, dim3, c='r', marker='o')
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='b', marker='o')
    ax.scatter(temp1, temp2, temp3, c='g', marker='o')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
              camera_direction[0], camera_direction[1], camera_direction[2], length=20, normalize=True)
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
              camera_up_direction[0], camera_up_direction[1], camera_up_direction[2], length=5, normalize=True, color='r')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
              camera_left_direction[0], camera_left_direction[1], camera_left_direction[2], length=5, normalize=True, color='g')
    plt.show()
    
    # save the data
    mdic = {'array':np.array([temp1, temp2, temp3]), 'wall':np.array([dim1, dim2, dim3]), 
            'camera_position':camera_position, 'camera_direction':camera_direction, 
            'camera_up_direction':camera_up_direction, 'camera_left_direction':camera_left_direction,
            'R':R, 'tvecs':tvecs, 'camera_matrix':camera_matrix}
    sio.savemat(output_path, mdic)