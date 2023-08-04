# SPAD_Array_Calibration

### Calibration procedure

1. Scan four vertices points using pf32.mlapp, the result is saved in `coords_voltages/vertices_voltage.mat`.
2. Prepare calibration video, which contains overall galvo's scanning procedure, and a chessboard on the wall.
3. Manually check the chessboard time, scanning start time, end time, and several path names, create a new json file at `setting/{video_name}.json`.
4. Run `python calibration_main.py`.

