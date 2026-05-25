# External libraries
import serial
import time
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import socket
from time import strftime

# Local libraries
import parameters
import particle_filter
import robot_python_code
import mppi

# The core robot class
class Robot:

    def __init__(self):
        self.connected_to_hardware = False
        self.running_trial = False
        self.extra_logging = False
        self.trial_start_time = 0
        self.msg_sender = None
        self.msg_receiver = None
        self.camera_sensor = robot_python_code.CameraSensor(parameters.camera_id)
        self.data_logger = robot_python_code.DataLogger(parameters.filename_start, parameters.data_name_list)
        self.robot_sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])
        self.camera_sensor_signal = [0,0,0,0,0,0]
        map = particle_filter.Map(parameters.wall_corner_list)
        self.particle_filter = particle_filter.ParticleFilter(
            parameters.num_particles,
            map, 
            particle_filter.State(0,0,0),
            particle_filter.State(1,1,1),
            True,
            0
        )

        self.mppi_controller = mppi.MPPI(
            goal_state=parameters.goal_state,
            goal_weight=parameters.goal_weight,
            K=parameters.mppi_K,
            T=parameters.mppi_T,
            alpha=parameters.mppi_alpha,
            delta_t=parameters.mppi_delta_t,
            nominal_encoder_rate=parameters.mppi_nominal_encoder_rate,
            sigma_steering=parameters.mppi_sigma_steering,
            sigma_encoder=parameters.mppi_sigma_encoder,
            heading_weight=parameters.mppi_heading_weight,
            path_weight=parameters.mppi_path_weight,
            wall_weight=parameters.mppi_wall_weight,
        )
        # MPPI is only activated after PF has a real fix and the user sets a goal.
        self.mppi_active = False
        self.last_mppi_trajectories = None  # cached for GUI visualisation

    @property
    def pf_std(self):
        """Mean std-dev (m) of particle x/y positions. Small value means PF has converged."""
        pts = np.array([[p.state.x, p.state.y]
                        for p in self.particle_filter.particle_set.particle_list])
        return float(np.mean(np.std(pts, axis=0)))

    def set_goal(self, goal_x, goal_y):
        """Set a new navigation goal and recompute the shortest-path reference from
        the current PF mean estimate. Only call once PF has converged."""
        goal = [float(goal_x), float(goal_y)]
        mean = self.particle_filter.particle_set.mean_state
        self.mppi_controller.set_goal(goal, start_state=[mean.x, mean.y])
        self.mppi_active = True

    # Create udp senders and receiver instances with the udp communication
    def setup_udp_connection(self, udp_communication):
        self.msg_sender = robot_python_code.MsgSender(time.perf_counter(), parameters.num_robot_control_signals, udp_communication)
        self.msg_receiver = robot_python_code.MsgReceiver(time.perf_counter(), parameters.num_robot_sensors, udp_communication)
        print("Reset msg_senders and receivers!")

    # Stop udp senders and receiver instances with the udp communication
    def eliminate_udp_connection(self):
        if self.msg_sender is not None:
            try:
                self.msg_sender.udp_communication.close()
            except Exception:
                pass
        self.msg_sender = None
        self.msg_receiver = None
        print("Eliminate UDP !!!")

    def update_state_estimate(self):
        u_t = np.array([self.robot_sensor_signal.encoder_counts, self.robot_sensor_signal.steering]) # robot_sensor_signal
        z_t = self.robot_sensor_signal
        delta_t = 0.1
        self.particle_filter.update(u_t, z_t, delta_t)

    # One iteration of the control loop to be called repeatedly
    def control_loop(self, cmd_speed=0, cmd_steering_angle=0, logging_switch_on=False):
        t0 = time.perf_counter()

        # Receive sensor data from the robot
        if self.msg_receiver is not None:
            self.robot_sensor_signal = self.msg_receiver.receive_robot_sensor_signal(self.robot_sensor_signal)
        t1 = time.perf_counter()

        # Update the state estimates from the particle filter
        self.update_state_estimate()
        t2 = time.perf_counter()

        # Build control signal: MPPI when active, else pass through manual commands
        if self.mppi_active:
            mean = self.particle_filter.particle_set.mean_state
            dist_to_goal = np.hypot(
                mean.x - self.mppi_controller.goal_state[0],
                mean.y - self.mppi_controller.goal_state[1]
            )
            if dist_to_goal <= parameters.goal_reached_dist:
                # Goal reached — stop the robot and deactivate MPPI
                self.mppi_active = False
                control_signal = [0, 0]
                print(f'[MPPI] Goal reached (dist={dist_to_goal:.3f} m) — stopping.')
            elif cmd_speed < 0:
                # Speed switch off — stop the robot.
                control_signal = [0, 0]
            else:
                U_nominal, trajectories = self.mppi_controller.mppi_iteration(
                    [mean.x, mean.y, mean.theta],
                    self.robot_sensor_signal.encoder_counts
                )
                self.last_mppi_trajectories = trajectories
                speed = cmd_speed if cmd_speed > 0 else parameters.mppi_nominal_speed_cmd
                control_signal = [speed, float(np.clip(U_nominal[0, 1], -20.0, 20.0))]
        else:
            control_signal = [max(0, cmd_speed), cmd_steering_angle]
        t3 = time.perf_counter()

        # Send control signal to the robot
        if self.msg_sender is not None:
            try:
                self.msg_sender.send_control_signal(control_signal)
            except OSError as e:
                print(f'UDP send error: {e.strerror}')
        t4 = time.perf_counter()

        total = (t4 - t0) * 1000
        if total > 10:  # print every tick to see MPPI cost
            print(f'[TIMING] recv={1000*(t1-t0):.1f}ms  pf={1000*(t2-t1):.1f}ms  '
                  f'mppi={1000*(t3-t2):.1f}ms  send={1000*(t4-t3):.1f}ms  '
                  f'TOTAL={total:.1f}ms  lidar_rays={self.robot_sensor_signal.num_lidar_rays}')

        # Log the data
        self.data_logger.log(logging_switch_on, time.perf_counter(), control_signal, self.robot_sensor_signal, self.particle_filter.particle_set.mean_state, self.particle_filter.particle_set)

