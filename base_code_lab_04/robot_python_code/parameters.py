# External libraries
import math
import numpy as np

# UDP parameters
localIP = "10.0.0.10" # Put your laptop computer's IP here
arduinoIP = "10.0.0.97" # Put your arduino's IP here
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 0
inches_to_meters = 0.0254
marker_length = 7 * inches_to_meters 
obj_points = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
camera_matrix = np.array([
        [
            1563.0532919672587,
            0.0,
            915.0889020362948
        ],
        [
            0.0,
            1560.5743446217068,
            563.6030469655277
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ], dtype=np.float32)

dist_coeffs = np.array(
  [
    -0.36492240253000724,
    0.03329766076723505,
    0.0009421722713071375,
    -2.9692152435985755e-06,
    0.08458318777758032
  ], dtype=np.float32)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100

# PF parameters, modify the map and num particles as you see fit.
num_particles = 100
distance_variance = 0.1  # meters^2, lidar measurement noise variance
# wall_corner_list = [
#     [0, 0, 2.74, 0], 
#     [0, 0, 0, 3.78], 
#     [0, 3.78, 1.92, 3.78],
#     [1.03, 1.61, 1.03, 2.19],
#     [1.03, 2.19, 1.41, 2.19],
#     [1.92, 3.78, 1.92, 3.32],
#     [1.92, 3.32, 2.74, 3.32],
#     [2.74, 3.32, 2.74, 0]
#     ]

wall_corner_list = [
    [-0.205, 2.858, -1.055, 2.858],
    [-1.055, 2.858, -1.33, 2.683],
    [-1.33, 2.683, -1.33, 2.183],
    [-1.33, 2.183, -1.53, 1.883],
    [-1.53, 1.883, -1.53, -0.717],
    [-1.53, -0.717, -1.405, -0.842],
    [-1.405, -0.842, -0.48, -0.917],
    [-0.48, -0.917, -0.48, -1.617],
    [-0.48, -1.617, -0.33, -1.767],
    [-0.33, -1.767, -0.28, -2.067],
    [-0.28, -2.067, -0.28, -3.167],
    [-0.28, -3.167, -0.155, -3.292],
    [-0.155, -3.292, 1.395, -3.292],
    [1.395, -3.292, 1.62, -3.117],
    [1.62, -3.117, 1.645, -1.592],
    [1.645, -1.592, 3.595, -1.492],
    [3.595, -1.492, 3.77, -1.267],
    [3.77, -1.267, 3.77, 1.283],
    [3.77, 1.283, 3.62, 1.983],
    [3.62, 1.983, 3.295, 2.258],
    [3.295, 2.258, 0.045, 2.258],
    [0.045, 2.258, -0.03, 2.683],
    [-0.03, 2.683, -0.205, 2.858],
]