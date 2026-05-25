# External libraries
import math
import numpy as np

# UDP parameters
localIP = "10.0.0.10" # Put your laptop computer's IP here
arduinoIP = "10.0.0.99" # Put your arduino's IP here
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
num_particles = 1000
distance_variance = 0.1  # meters^2, lidar measurement noise variance
lidar_ray_step = 1       # use every Nth lidar ray in weight calculation (1 = all rays)
use_obstacle_buffer = False   # True: apply a fixed penalty when measured << expected (obstacle blocks ray)
                             # False: treat all rays with the same Gaussian likelihood
obstacle_buffer_threshold = 0.3   # meters; if (expected - measured) > this, ray is treated as blocked
obstacle_buffer_penalty   = -0.3  # log-weight penalty applied to blocked rays (negative)
reinit_weight_threshold   = 1e-9 # max particle weight below which a cycle counts as "lost"
reinit_cycles_required    = 5    # number of consecutive lost cycles before reinitializing uniformly
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
    [0.395, 1.795, -0.805, 1.745],
    [-0.805, 1.745, -0.93, 1.62],
    [-0.93, 1.62, -1.005, 0.845],
    [-1.005, 0.845, -3.305, 0.845],
    [-3.305, 0.845, -3.48, 0.57],
    [-3.48, 0.57, -3.48, -0.23],
    [-3.48, -0.23, -3.43, -1.08],
    [-3.43, -1.08, -3.205, -1.455],
    [-3.205, -1.455, -1.13, -1.53],
    [-1.13, -1.53, -1.13, -2.03],
    [-1.13, -2.03, -0.98, -2.18],
    [-0.98, -2.18, -0.98, -3.18],
    [-0.98, -3.18, -0.855, -3.305],
    [-0.855, -3.305, 0.645, -3.455],
    [0.645, -3.455, 0.77, -3.33],
    [0.77, -3.33, 0.795, -3.005],
    [0.795, -3.005, 1.395, -2.905],
    [1.395, -2.905, 1.545, -2.655],
    [1.545, -2.655, 1.945, -2.655],
    [1.945, -2.655, 2.07, -2.53],
    [2.07, -2.53, 2.17, -2.28],
    [2.17, -2.28, 2.12, -0.08],
    [2.12, -0.08, 2.645, -0.055],
    [2.645, -0.055, 2.77, 0.07],
    [2.77, 0.07, 2.87, 0.62],
    [2.87, 0.62, 2.77, 0.72],
    [2.77, 0.72, 2.77, 1.17],
    [2.77, 1.17, 2.645, 1.295],
    [2.645, 1.295, 1.845, 1.295],
    [1.845, 1.295, 1.82, 1.52],
    [1.82, 1.52, 1.495, 1.745],
    [1.495, 1.745, 0.395, 1.795],
]

# MPPI parameters
goal_state = [0, 0]
goal_weight = 100
goal_reached_dist = 0.15  # metres — stop MPPI when robot is within this distance of the goal
mppi_K = 100
mppi_T = 60          # longer horizon so a U-turn fits inside the planning window
mppi_alpha = 0.1
mppi_delta_t = 0.1
# Mean encoder increment per delta_t=0.1s step derived from lab 2 straight-line trials
# (9 runs at speed=70, steering=0):
#
#   File                                      Duration   Mean enc/step
#   robot_data_70_0_12_02_26_19_09_16.pkl     5000 ms    79.48
#   robot_data_70_0_12_02_26_19_12_32.pkl     5000 ms    80.58
#   robot_data_70_0_12_02_26_19_14_16.pkl     5000 ms    81.49
#   robot_data_70_0_12_02_26_19_16_32.pkl    10000 ms    87.41
#   robot_data_70_0_12_02_26_19_18_50.pkl    10000 ms    88.72
#   robot_data_70_0_12_02_26_19_21_16.pkl    10000 ms    86.61
#   robot_data_70_0_12_02_26_19_23_39.pkl    15000 ms    88.99
#   robot_data_70_0_12_02_26_19_26_05.pkl    15000 ms    88.08
#   robot_data_70_0_12_02_26_19_27_27.pkl    15000 ms    88.10
#
#   Overall mean=85.5, median=87.4, std=3.6 counts/step
mppi_nominal_encoder_rate = 85.5
# Exploration noise for MPPI trajectory sampling.
# sigma_steering: std-dev of per-step steering perturbation in servo degrees.
#   The servo range is ±20°; ~6° gives good trajectory diversity without
#   sampling into walls on every step.
# sigma_encoder: std-dev of per-step encoder increment perturbation (counts).
#   Set to 0 to keep all trajectories at the same nominal speed.
mppi_sigma_steering = 10.0  # wider — ~16% of samples draw near max steering (20°)
mppi_sigma_encoder  = 0.0
# Speed command (slider units, 0-100) that produced the nominal_encoder_rate data above.
# Used automatically when MPPI is active so the user doesn't need to set the slider.
mppi_nominal_speed_cmd = 70
# Heading alignment weight: penalises each trajectory step's heading mismatch vs path tangent.
# Critical for recovering when robot faces away from the goal.
# 1.5 ≈ 1.5 rad (~86°) of heading error costs the same as 1.5 m of path deviation.
mppi_heading_weight = 1.5
mppi_path_weight   = 1.0   # cost per metre of deviation from the Dijkstra reference path, summed over T
mppi_wall_weight   = 50.0  # at 0.5 m from wall: 50*exp(-1)=18.4 per step; at 0.1 m: 50*exp(-0.2)=40.9 per step
