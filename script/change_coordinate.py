import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


array_data = sio.loadmat('SPAD_array_cali/array_data_C0016.mat')
output_path = 'SPAD_array_cali/relative_position.mat'

array = array_data['array']
wall = array_data['wall']
camera_position = array_data['camera_position']
camera_position = np.squeeze(camera_position)
camera_up_direction = array_data['camera_up_direction']
camera_direction = array_data['camera_direction']
R = array_data['R']
inverse_R = np.linalg.inv(R)
tvecs = array_data['tvecs']
tvecs = np.squeeze(tvecs)

array = array.T - tvecs
wall = wall.T - tvecs
camera_position = camera_position - tvecs
camera_position = camera_position.T
print(array.shape, inverse_R.shape)
array = array @ R
wall = wall @ R
camera_position = camera_position.T @ R
camera_up_direction = (camera_up_direction.T @ R)[0]
camera_direction = (camera_direction.T @ R)[0]

array_x = array[1] - array[0]
array_y = array[16] - array[0]
array_z = np.cross(array_y, array_x)

# draw in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array[:, 0], array[:, 1], array[:, 2], c='b', marker='o')
ax.scatter(wall[:, 0], wall[:, 1], wall[:, 2], c='g', marker='o')
print(camera_position)
ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o', s=100)
ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
            camera_up_direction[0], camera_up_direction[1], camera_up_direction[2], length=5, normalize=True, color='r')
ax.quiver(camera_position[0], camera_position[1], camera_position[2],
            camera_direction[0], camera_direction[1], camera_direction[2], length=15, normalize=True, color='b')
ax.quiver(array[0, 0], array[0, 1], array[0, 2],
           array_z[0], array_z[1], array_z[2], length=15, normalize=True, color='g')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# let the axis have the same scale
ax.set_xlim(-80, 40)
ax.set_ylim(-60, 60)
ax.set_zlim(0, 120)
plt.show()

mdic = {'array':array, 'array_normal':array_z, 'wall':wall}
sio.savemat(output_path, mdic)