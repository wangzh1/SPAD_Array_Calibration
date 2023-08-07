import argparse
from SPAD_calibration import *
from tqdm import tqdm

# A json represents a wall, so in the following content, json_name == wall_name
parser = argparse.ArgumentParser(description='Calibration Type (init/pixelcali/meas)')
parser.add_argument('json_name', type=str, default=None, help='Filename of json, representing a wall/setting.')
parser.add_argument('type', type=str, default='cali', help='Program type, init/pixelcali/meas/galvo_optimize.')
parser.add_argument('meas_name', type=str, default=None, help='Folder name of measure.')

args = parser.parse_args()

if args.type == 'init':
    """
    options: Template matching or not, throw bad points or not.
    returns: Bijection MLP saved under 'models/'.
    """

    json_path = 'setting/' + args.json_name + '.json'

    # Create cali object
    cali = SPAD_calibration(args.json_name)

    # Let user choose whether to do video template matching
    if_template_matching = input("Do you want do template matching first? [y/n]: ")

    if if_template_matching == 'y':
        cali.template_matching() # output detected_pixels.mat

    # Get the parameters of camera and the scene
    camera_matrix, rvecs, tvecs = cali.calibrate_scene(cali.wall_board_time)

    R, _ = cv2.Rodrigues(rvecs[0])
    t = tvecs[0]

    input_params = cali.voltage_cali
    voltage_cali = np.concatenate((input_params['voltage_x'], input_params['voltage_y']), axis=0)

    # Deal with bad point!
    good_points_num = int(input("Number of tail good points: "))

    input_voltages = np.concatenate((cali.voltage_cali['voltage_x'], cali.voltage_cali['voltage_y']), axis=0).T[-good_points_num:]
    detected_pixels = scipy.io.loadmat(cali.detected_pixels_path)['detected_pixels'][-good_points_num:] # (num_of_points, 2)

    # Prepare training dataset for MLP
    coords_of_voltages = cali.img_to_world(detected_pixels, 
                                           rvecs, 
                                           tvecs, 
                                           camera_matrix).T # (num_of_points, 2)

    model_v2c = Model(2,7,2)
    model_c2v = Model(2,7,2)

    data_v2c = MyData(input_voltages, coords_of_voltages) # (num_of_points, 2)
    data_c2v = MyData(coords_of_voltages, input_voltages)

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

    # Start training
    for epoch in tqdm(range(epochs)):
        sum_loss = 0
        train_correct = 0
        for data in data_loader_v2c:
            inputs, labels = data # inputs: (num_of_points, 2)
            inputs = inputs.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.float32)
            outputs = model_v2c(inputs) # outputs: (num_of_points, 2)

            optimizer_v2c.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer_v2c.step()
            sum_loss += loss.item()

        if epoch % 1000 == 0:
            print('Voltage2Coords model training: epoch: {}, loss: {}'.format(epoch, round(sum_loss, 5)))

    for epoch in tqdm(range(epochs)):
        sum_loss = 0
        train_correct = 0
        for data in data_loader_c2v:
            inputs, labels = data # inputs: (num_of_points, 2)
            inputs = inputs.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.float32)
            outputs = model_c2v(inputs) # outputs: (num_of_points, 2)

            optimizer_c2v.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer_c2v.step()
            sum_loss += loss.item()

        if epoch % 1000 == 0:
            print('Coords2Voltage model training: epoch: {}, loss: {}'.format(epoch, sum_loss))

    # Save models
    torch.save(model_v2c, 'models/'+args.json_name+'_v2c.pt')
    torch.save(model_c2v, 'models/'+args.json_name+'_c2v.pt')
    print("Successfully saved models!")

elif args.type == 'pixelcali':
    """
    vertex_voltages.mat -> all_pixels_coordinates.mat -> voltage_x, voltage_y(2, 32, 32)
    Lens distortion not being considered, cannot function effectively.
    """
    f = float(input("Focal length is (mm): "))
    # Define SPAD Array's camera matrix
    dx = dy = 1.6 / 32
    fx, fy = f/dx, f/dy
    u0 = v0 = 15.5
    spad_array_matrix = np.array([[fx,  0, u0], 
                                  [ 0, fy, v0], 
                                  [ 0,  0,  1]])
    
    model_v2c = torch.load('models/'+args.json_name+'_v2c.pt')
    model_c2v = torch.load('models/'+args.json_name+'_c2v.pt')
    
    # Shape = (2, 4).T -> (4, 2) 
    # Order: (0, 0) -> (31, 0) -> (31, 31) -> (0, 31)
    vertices_voltage = torch.tensor(scipy.io.loadmat('transforms/'+args.json_name+'.mat')['vertices_voltage'].T, 
                                    dtype=torch.float32)
    vertices_coords = model_v2c(vertices_voltage).detach().numpy()
    vertices_coords = np.concatenate((vertices_coords, np.zeros([4, 1])), axis=1, dtype=np.float64)
    vertices_index = np.array([[ 0,  0],
                               [31,  0],
                               [31, 31],
                               [ 0, 31]], dtype=np.float64)

    _, rvecs, t =cv2.solvePnP(objectPoints=vertices_coords, 
                              imagePoints=vertices_index, 
                              cameraMatrix=spad_array_matrix,
                              distCoeffs=None,
                              flags=cv2.SOLVEPNP_P3P)
    
    R, _ = cv2.Rodrigues(rvecs) # R.shape=(3, 3), t.shape=(1, 3)

    img_pixels = np.mgrid[0:32, 0:32].reshape(2, -1) 
    img_pixels_expanded = np.vstack((img_pixels, np.ones([1, 1024]))) # (3, 1024)

    scale = (np.linalg.inv(R) @ t)[2, :] / \
            (np.linalg.inv(R) @ np.linalg.inv(spad_array_matrix) @ img_pixels_expanded)[2, :] # (1, 1024)
        
    pixels_coords = scale * (np.linalg.inv(R) @ np.linalg.inv(spad_array_matrix) @ img_pixels_expanded) - np.linalg.inv(R) @ t # (3, 1024)
    pixels_coords = pixels_coords[:2] # (2, 1024)
    pixels_coords_tensor = torch.tensor(pixels_coords.T, dtype=torch.float32) # (1024, 2)
    pixels_voltage = model_c2v(pixels_coords_tensor).detach().numpy().T.reshape(2, 32, 32)
    pixels_coords = pixels_coords.reshape(2, 32, 32)

    scipy.io.savemat('transforms/'+args.json_name+'_pixels_cali_result.mat', 
                     {'coords': pixels_coords, 'voltage_x': pixels_voltage[0], 'voltage_y': pixels_voltage[1]})
    
    inferred_vertices_coords = np.vstack((pixels_coords[:, 0, 0],
                                          pixels_coords[:, 31, 0],
                                          pixels_coords[:, 31, 31],
                                          pixels_coords[:, 0, 31])) # (4, 2)
    
    inferred_vertices_coords_normalized = inferred_vertices_coords / np.max(inferred_vertices_coords, axis=0)
    vertex_coords_normalized = vertices_coords[:, :2] / np.max(vertices_coords[:, :2], axis=0)

    print("Successfully calibrated pixels! Vertex loss: {}.".format(np.mean((inferred_vertices_coords_normalized - vertex_coords_normalized) ** 2)))

elif args.type == 'meas':
    """
    1. 'WallName_voltage_grid_interpolated.mat' -> 'resource/PF32_workspace/MeasName/coords_receive.mat' 
    2. 'MeasName_pre_meas.mat' -> 'resource/input/input_params.mat', 'resource/PF32_workspace/MeasName/coords_laser.mat'
    3. Optimize galvanometer's coordinates, save under 'transforms/WallName_coords_galva.mat'
    """

    # Voltage grid -> coords_receive
    model_v2c = torch.load('models/'+args.json_name+'_v2c.pt')
    voltage_grid = scipy.io.loadmat('transforms/'+args.json_name+'_voltage_grid_interpolated.mat') 
    voltage_x = voltage_grid['voltage_x'] # (1, 1024)
    voltage_y = voltage_grid['voltage_y']
    voltages_receive = np.vstack((voltage_x, voltage_y)).T # (1024, 2)
    
    voltages_receive_tensor = torch.tensor(voltages_receive, dtype=torch.float32)
    coords_receive = model_v2c(voltages_receive_tensor).detach().numpy() # (1024, 2)
    scipy.io.savemat('C:/Users/74338/Desktop/IQI/resources/PF32_workspace/'+args.meas_name+'/positions_receive.mat', {'positions_receive': coords_receive.T})

    # Measure prepare
    model_c2v = torch.load('models/'+args.json_name+'_c2v.pt')
    pre_meas = scipy.io.loadmat('transforms/'+args.meas_name+'_pre_meas.mat')
    exposure = float(pre_meas['exposure'])
    edge_num = int(pre_meas['edge_num'])
    voltage_x_origin = pre_meas['voltage_x_origin']
    voltage_y_origin = pre_meas['voltage_y_origin']
    scale = np.squeeze(pre_meas['scale'])

    voltage_origin = np.hstack((voltage_x_origin, voltage_y_origin))
    voltages_origin_tensor = torch.tensor(voltage_origin, dtype=torch.float32)
    coords_origin = model_v2c(voltages_origin_tensor).detach().numpy() # (1, 2)

    coords_left_top = coords_origin + np.array([-scale/2, scale/2])
    coords_right_bottom = coords_origin + np.array([scale/2, -scale/2])


    x, y = np.meshgrid(np.linspace(coords_left_top[0,0],coords_right_bottom[0,0],edge_num), 
                       np.linspace(coords_left_top[0,1],coords_right_bottom[0,1],edge_num)) # (edge_num, edge_num)   
    
    x = x.reshape(edge_num ** 2, 1)
    y = y.reshape(edge_num ** 2, 1)
    coords_laser = np.hstack((x, y))
    coords_laser_tensor = torch.tensor(coords_laser, dtype=torch.float32)

    voltage_laser = model_c2v(coords_laser_tensor).detach().numpy().astype('float64') # (edge_num**2, 2)

    scipy.io.savemat('C:/Users/74338/Desktop/IQI/resources/input/input_params_meas.mat', 
                     {'voltage_x': voltage_laser[:, 0].T, 'voltage_y': voltage_laser[:, 1].T, 'exposure':exposure})
    scipy.io.savemat('C:/Users/74338/Desktop/IQI/resources/PF32_workspace/'+args.meas_name+'/positions_laser.mat',
                     {'positions_laser': coords_laser.T})

    print('Okay now you can start measurement.')

elif args.type == 'galvo_optimize':
    
    model_v2c = torch.load('models/'+args.json_name+'_v2c.pt')
    input_params = scipy.io.loadmat('input_params/'+args.json_name+'.mat')
    voltage = np.vstack((input_params['voltage_x'], input_params['voltage_y'])).T # (num_of_points, 2)
    voltage_tensor = torch.tensor(voltage, dtype=torch.float32)
    coords = model_v2c(voltage_tensor).detach().numpy().astype('float64')
    coords = np.hstack((coords, np.zeros([coords.shape[0], 1])))
    coords_galva = initilization(coords, voltage).reshape(3, 1)

    scipy.io.savemat('transforms/'+args.json_name+'_galva.mat', {'coordinates_galva': coords_galva})
else:
    print('No such mode! Please input one of "init/meas".')
