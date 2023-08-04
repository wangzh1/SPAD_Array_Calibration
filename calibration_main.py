import argparse
from SPAD_calibration import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calibration Type (cali/infer)')
parser.add_argument('json_name', type=str, default=None, help='Filename of json')
parser.add_argument('type', type=str, default='cali', help='Program type, cali/infer')

args = parser.parse_args()

json_path = 'setting/' + args.json_name + '.json'
model_path = 'models/'
cali = SPAD_calibration(args.json_name)
# cali.template_matching() # output detected_pixels.mat

camera_matrix_wall, rvecs_wall, tvecs_wall = cali.calibrate_scene(cali.wall_board_time)
camera_matrix_spad, rvecs_spad, tvecs_spad = cali.calibrate_scene(cali.spad_board_time)

R_wall, _ = cv2.Rodrigues(rvecs_wall[0])
t_wall = tvecs_wall[0]
R_spad, _ = cv2.Rodrigues(rvecs_spad[0])
t_spad = tvecs_spad[0]

# input_params = scipy.io.loadmat(cali.voltage_cali_path)
# voltage_cali = np.concatenate(input_params['voltage_x'], input_params['voltage_y'], axis=0)

detected_pixels = scipy.io.loadmat(cali.detected_pixels_path)['detected_pixels'][13:] # (num_of_points, 2)

coords_of_voltages = cali.img_to_world(detected_pixels, 
                                       rvecs_wall, 
                                       tvecs_wall, 
                                       camera_matrix_wall) # (2, num_of_points)

model_v2c = cali.Model(2,7,2)
model_c2v = cali.Model(2,7,2)

data_v2c = cali.Mydata(np.concatenate((cali.voltage_cali['voltage_x'], 
                                       cali.voltage_cali['voltage_y']), axis=0).T[13:], 
                       detected_pixels)

data_c2v = cali.Mydata(detected_pixels,
                       np.concatenate((cali.voltage_cali['voltage_x'], 
                                       cali.voltage_cali['voltage_y']), axis=0).T[13:])

data_loader_v2c = torch.utils.data.DataLoader(dataset = data_v2c,
                                              batch_size = 32, 
                                              shuffle = True)

data_loader_c2v = torch.utils.data.DataLoader(dataset = data_c2v,
                                              batch_size = 32, 
                                              shuffle = True)

loss_function = torch.nn.MSELoss()
optimizer_v2c = torch.optim.Adam(model_v2c.parameters())
optimizer_c2v = torch.optim.Adam(model_c2v.parameters())

epochs = 5000

for epoch in tqdm(range(epochs)):
    
    sum_loss=0
    train_correct=0

    for data in data_loader_v2c:
        inputs, labels = data # inputs: [100,2]
        inputs_new = torch.tensor(inputs, dtype=torch.float32)
        labels_new = torch.tensor(labels, dtype=torch.float32)
        outputs=model_v2c(inputs_new) # outputs: [100,2]

        optimizer_v2c.zero_grad()
        loss = loss_function(outputs,labels_new)
        loss.backward()
        optimizer_v2c.step()
        sum_loss += loss.item()

    if epoch % 1000 == 0:
        print('Voltage2Coords model training: epoch: {}, loss: {}'.format(epoch, sum_loss))

for epoch in tqdm(range(epochs)):
    
    sum_loss=0
    train_correct=0

    for data in data_loader_c2v:
        inputs, labels = data # inputs: [100,2]
        inputs_new = torch.tensor(inputs, dtype=torch.float32)
        labels_new = torch.tensor(labels, dtype=torch.float32)
        outputs=model_c2v(inputs_new) # outputs: [100,2]

        optimizer_c2v.zero_grad()
        loss = loss_function(outputs,labels_new)
        loss.backward()
        optimizer_c2v.step()
        sum_loss += loss.item()

    if epoch % 1000 == 0:
        print('Coords2Voltage model training: epoch: {}, loss: {}'.format(epoch, sum_loss))

