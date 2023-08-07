# SPAD_Array_Calibration

This repo is for non-confocal SPAD array system calibration.

### Environment setup

```shell
pip install -r requirements.txt
```

### Calibration procedure in ShanghaiTech

1. Press pf32.mlapp livestraming button, and control the galvanometer voltage manually to find the voltages corresponding to four corners of PF32 frame. Press 'set' in sequence, then press 'scan four vertices'. After scanning, press the 'show and save voltage grid' and scan the full grid.

2. Prepare calibration video, which contains overall galvanometer's scanning procedure and a chessboard on the wall. Note that the scanning area should include the whole measure area, for MLP's better prediction.

3. Manually check the chessboard time, scanning start time, end time, and several path names, create a new json file at `setting/{json_name}.json`.

4. Run:

   ```shell
   python calibration_main.py {json_name} init
   ```

   This script does:

   - Read the frame at **template time**, you need to manually select the template for matching.
   - Read the whole video with `interval` set in the class `SPAD_calibration`'s `init` function, and match frames with template, output index of pixels of the laser point controlled by the galvanometer.
   - Train two MLPs: voltage to point & point to voltage.
   - Save the two models in `models/`.

5. Turn the full voltage grid into positions using pretrained pytorch model.

6. In order to generate the voltage_x and voltage_y for measurement, you can use:

   ```shell
   python calibration_main.py {json_name} meas {meas_folder_name}
   ```

   It does:

   -  'WallName_voltage_grid_interpolated.mat' -> 'resource/PF32_workspace/MeasName/coords_receive.mat' 
   - 'MeasName_pre_meas.mat' -> 'resource/input/input_params.mat', 'resource/PF32_workspace/MeasName/coords_laser.mat'
   - Optimize galvanometer's coordinates, save under 'transforms/WallName_coords_galva.mat'

