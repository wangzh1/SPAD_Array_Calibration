'''
Calibration part1: template matching
find all the dots in the video and save the result in a mat file

reminder: the format of the json file:
    "video_path": indicates the path of the video, absolute path is recommended
    "template_time": the time of the template frame (in seconds)
    "template_x(y)_1(2)": the position of the template frame (in pixels)
    "chessboard_time": the time of the chess board frame (in seconds)
    "begin_time": the begin time of the video (in seconds)
    "end_time": the end time of the video (in seconds)
'''
import cv2
import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


# read the json file
json_path = 'setting/C0016.json'
output_path = 'detection_result/detection_result_C0016.mat'

# parameters
with open(json_path, 'r') as f:
    content = json.load(f)
    
interval = 3
threshold = 0.7
template_width = 40
video_path = content['video_path']
template_time = content['template_time']
begin_time = content["begin_time"]
end_time = content["end_time"]


def line_select_callback(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    global glob_x1, glob_x2, glob_y1, glob_y2
    glob_x1, glob_x2, glob_y1, glob_y2 = y1, y2, x1, x2
    print("Center: ", (glob_x1 + glob_x2) // 2, ",", (glob_y1 + glob_y2) // 2)

def toggle_selector(event):
    pass

# get the template frame
def get_template_frame(video, time, x_1, x_2, y_1, y_2):
    if not video.isOpened():
        print("Could not open video file")  
    video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    ret, frame = video.read()
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
    toggle_selector.RS = RectangleSelector(
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

if __name__ == "__main__":
    # read the video
    video = cv2.VideoCapture(video_path)
    
    # get template frame
    center_x, center_y = find_center(video, template_time, content)
    x_1, x_2, y_1, y_2 = center_x - template_width // 2, center_x + template_width // 2, center_y - template_width // 2, center_y + template_width // 2
    template = get_template_frame(video, template_time, x_1, x_2, y_1, y_2)
    # show the template frame
    plt.imshow(template)
    plt.show()
    
    # get the points
    points = get_points(video, begin_time*1000, end_time*1000, interval, threshold, template)
    
    # cluster the points
    output_point = cluster_points(points)

    plt.figure()
    for dot_pos in output_point:
        plt.scatter(dot_pos[0], dot_pos[1], c='r', s=10)
    print(len(output_point))
    mdic = {'pt':output_point}
    sio.savemat(output_path, mdic)
    plt.savefig('detection_result/detection_result.png')
    plt.clf()
    plt.imshow(plt.imread('detection_result/detection_result.png'))
    plt.show()
    