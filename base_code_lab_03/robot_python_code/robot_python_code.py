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


# Function to try to connect to the robot via udp over wifi
def create_udp_communication(arduinoIP, localIP, arduinoPort, localPort, bufferSize):
    try:
        udp = UDPCommunication(arduinoIP, localIP, arduinoPort, localPort, bufferSize)
        print("Success in creating udp communication")
        return udp, True
    except:
        print("Failed to create udp communication!")
        return _, False
        
        
# Class to hold the UPD over wifi connection setup
class UDPCommunication:
    def __init__(self, arduinoIP, localIP, arduinoPort, localPort, bufferSize):
        self.arduinoIP = arduinoIP
        self.arduinoPort = arduinoPort
        self.localIP = localIP
        self.localPort = localPort
        self.bufferSize = bufferSize
        self.UDPServerSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
        self.UDPServerSocket.bind((localIP, localPort))
        
    # Receive a message from the robot
    def receive_msg(self):
        bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        clientMsg = "{}".format(message.decode())
        clientIP = "{}".format(address)
        
        return clientMsg
       
    # Send a message to the robot
    def send_msg(self, msg):
        bytesToSend = str.encode(msg)
        self.UDPServerSocket.sendto(bytesToSend, (self.arduinoIP, self.arduinoPort))


# Class to hold the data logger that records data when needed
class DataLogger:

    # Constructor
    def __init__(self, filename_start, data_name_list):
        self.filename_start = filename_start
        self.filename = filename_start
        self.line_count = 0
        self.dictionary = {}
        self.data_name_list = data_name_list
        for name in data_name_list:
            self.dictionary[name] = []
        self.currently_logging = False

    # Open the log file
    def reset_logfile(self, control_signal):
        self.filename = self.filename_start + "_"+str(control_signal[0])+"_"+str(control_signal[1]) + strftime("_%d_%m_%y_%H_%M_%S.pkl")
        self.dictionary = {}
        for name in self.data_name_list:
            self.dictionary[name] = []

        
    # Log one time step of data
    def log(self, logging_switch_on, time, control_signal, robot_sensor_signal, camera_sensor_signal, state_mean, state_covariance):
        if not logging_switch_on:
            if self.currently_logging:
                self.currently_logging = False
        else:
            if not self.currently_logging:
                self.currently_logging = True
                self.reset_logfile(control_signal)

        if self.currently_logging:
            self.dictionary['time'].append(time)
            self.dictionary['control_signal'].append(control_signal)
            self.dictionary['robot_sensor_signal'].append(robot_sensor_signal)
            self.dictionary['camera_sensor_signal'].append(camera_sensor_signal)
            self.dictionary['state_mean'].append(state_mean)
            self.dictionary['state_covariance'].append(state_covariance)

            self.line_count += 1
            if self.line_count > parameters.max_num_lines_before_write:
                self.line_count = 0
                with open(self.filename, 'wb') as file_handle:
                    pickle.dump(self.dictionary, file_handle)

# Utility for loading saved data
class DataLoader:

    # Constructor
    def __init__(self, filename):
        self.filename = filename
        
    # Load a dictionary from file.
    def load(self):
        with open(self.filename, 'rb') as file_handle:
            loaded_dict = pickle.load(file_handle)
        return loaded_dict

# Class to hold a message sender
class MsgSender:

    # Time step size between message to robot sends, in seconds
    delta_send_time = 0.1

    # Constructor
    def __init__(self, last_send_time, msg_size, udp_communication):
        self.last_send_time = last_send_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
        
    # Pack and send a control signal to the robot.
    def send_control_signal(self, control_signal):
        packed_send_msg = self.pack_msg(control_signal)
        self.send(packed_send_msg)
    
    # If its time, send the control signal to the robot.
    def send(self, msg):
        new_send_time = time.perf_counter()
        if new_send_time - self.last_send_time > self.delta_send_time:
            message = ""
            for data in msg:
                message = message + str(data)
            self.udp_communication.send_msg(message)
            self.last_send_time = new_send_time
      
    # Pack a message so it is in the correct format for the robot to receive it.
    def pack_msg(self, msg):
        packed_msg = ""
        for data in msg:
            if packed_msg == "":
                packed_msg = packed_msg + str(data)
            else:
                packed_msg = packed_msg + ", "+ str(data)
        packed_msg = packed_msg + "\n"
        return packed_msg
        
# The robot's message receiver
class MsgReceiver:

    # Determines how often to look for incoming data from the robot.
    delta_receive_time = 0.05

    # Constructor
    def __init__(self, last_receive_time, msg_size, udp_communication):
        self.last_receive_time = last_receive_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
      
    # Check if its time to look for a new message from the robot.
    def receive(self):
        new_receive_time = time.perf_counter()
        if new_receive_time - self.last_receive_time > self.delta_receive_time:
            received_msg = self.udp_communication.receive_msg()
            self.last_receive_time = new_receive_time
            return True, received_msg
            
        return False, ""
    
    # Given a new message, put it in a digestable format
    def unpack_msg(self, packed_msg):
        unpacked_msg = []
        msg_list = packed_msg.split(',')
        if len(msg_list) >= self.msg_size:
            for data in msg_list:
                unpacked_msg.append(float(data))
            return True, unpacked_msg

        return False, unpacked_msg
        
    # Check for new message and unpack it if there is one.
    def receive_robot_sensor_signal(self, last_robot_sensor_signal):
        robot_sensor_signal = last_robot_sensor_signal
        receive_ret, packed_receive_msg = self.receive()
        if receive_ret:
            unpack_ret, unpacked_receive_msg = self.unpack_msg(packed_receive_msg)
            if unpack_ret:
                robot_sensor_signal = RobotSensorSignal(unpacked_receive_msg)
            
        return robot_sensor_signal

# Class to hold a camera sensor data. Not needed for lab 1.
class CameraSensor:

    # Constructor
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.inch_to_meter = 0.0254

        # Marker length for world frame
        self.w_marker_length = (5 + 12/16) * self.inch_to_meter 
        self.w_obj_points = self.get_obj_points(self.w_marker_length)

        # Marker length for robot
        self.r_marker_length = (3 + 13/16) * self.inch_to_meter
        self.r_obj_points = self.get_obj_points(self.r_marker_length)

        self.world_marker_ids = [0, 2, 1]

    def get_obj_points(self, marker_length):
        return np.array([[-marker_length/2, marker_length/2, 0],
                        [marker_length/2, marker_length/2, 0],
                        [marker_length/2, -marker_length/2, 0],
                        [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)

    # Get a new pose estimate from a camera image
    def get_signal(self, last_camera_signal):
        # x_tm1, y_tm1, theta_tm1 = last_camera_signal
        # ret, pose_estimate = self.get_pose_estimate()
        # if ret:
        #     x_t, y_t, theta_t = pose_estimate
        #     # If camera signal is not very different from last signal, return the new signal. Otherwise, return the last signal to avoid noise.
        #     if abs(x_t - x_tm1) < 0.5 and abs(y_t - y_tm1) < 0.5 and abs(theta_t - theta_tm1) < np.pi/4:    
        #         return pose_estimate
        #     else:
        #         return last_camera_signal
        ret, pose_estimate = self.get_pose_estimate()
        if ret:
            return pose_estimate
        return None
        
    # If there is a new image, calculate a pose estimate from the fiducial tag on the robot.
    def get_pose_estimate(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, []

        frame = cv2.undistort(frame, parameters.camera_matrix, parameters.dist_coeffs)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(frame)

        if ids is not None:
            ids = ids.flatten()
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Marker ids 0, 1, 3 define world coordinates
            world_coordinates = {}
            world_corners = {}
            for i in range(len(ids)):
                if ids[i] in self.world_marker_ids:
                    # Use solvePnP with all four corners of the marker
                    success, rvec, tvec = cv2.solvePnP(self.w_obj_points, corners[i], parameters.camera_matrix, parameters.dist_coeffs)
                    if success:
                        # Use the first corner of the detected marker for world coordinates
                        # corners[i][0][0] is the first corner (x, y) in image coordinates
                        # Project the first corner to world coordinates using the transformation
                        # tvec is the translation vector for the marker center, but we want the first corner
                        # Transform the first corner from object points to camera coordinates
                        first_corner_obj = self.w_obj_points[0].reshape(1, 3)
                        first_corner_cam, _ = cv2.projectPoints(first_corner_obj, rvec, tvec, parameters.camera_matrix, parameters.dist_coeffs)
                        # For world coordinates, use solvePnP result and add the offset from the marker center to the first corner
                        # But since solvePnP gives the pose for the marker, we can transform the first corner using the pose
                        R, _ = cv2.Rodrigues(rvec)
                        first_corner_world = R @ self.w_obj_points[0] + tvec.flatten()
                        world_coordinates[ids[i]] = first_corner_world
                        world_corners[ids[i]] = corners[i][0][0]

            if all(marker in world_coordinates for marker in self.world_marker_ids):
                # Define the world frame transformation using the first corners
                origin = world_coordinates[0]
                x_axis = world_coordinates[2] - origin  # 0 to 2 is x-axis
                y_axis = world_coordinates[1] - origin  # 0 to 1 is y-axis

                # Orthogonalize and normalize axes
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                proj = np.dot(y_axis, x_axis) * x_axis
                y_axis = y_axis - proj

                y_axis = y_axis / np.linalg.norm(y_axis)
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / np.linalg.norm(z_axis)

                # Construct rotation matrix
                rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
                translation_vector = origin

                # Create a 4x4 homogeneous transformation matrix
                world_to_camera_transform = np.eye(4)
                world_to_camera_transform[:3, :3] = rotation_matrix
                world_to_camera_transform[:3, 3] = translation_vector

            for i in range(len(ids)):
                if ids[i] not in self.world_marker_ids and all(marker in world_coordinates for marker in self.world_marker_ids):
                    success, rvec, tvec = cv2.solvePnP(self.r_obj_points, corners[i], parameters.camera_matrix, parameters.dist_coeffs)
                    if success:
                        # Transform rvec and tvec to world coordinates
                        camera_to_marker_transform = np.eye(4)
                        camera_to_marker_transform[:3, :3] = cv2.Rodrigues(rvec)[0]
                        camera_to_marker_transform[:3, 3] = tvec.flatten()

                        marker_in_world = np.dot(np.linalg.inv(world_to_camera_transform), camera_to_marker_transform)

                        x_world = marker_in_world[0, 3]
                        y_world = marker_in_world[1, 3]
                        theta_world = np.arctan2(marker_in_world[1, 0], marker_in_world[0, 0])

                        pose_estimate = np.array([x_world, y_world, theta_world])
                        return True, pose_estimate

        return False, []
    
    # Close the camera stream
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# A storage vessel for an instance of a robot signal
class RobotSensorSignal:

    # Constructor
    def __init__(self, unpacked_msg):
        self.encoder_counts = int(unpacked_msg[0])
        self.steering = int(unpacked_msg[1])
        self.num_lidar_rays = int(unpacked_msg[2])
        self.angles = []
        self.distances = []
        for i in range(self.num_lidar_rays):
            index = 3 + i*2
            self.angles.append(unpacked_msg[index])
            self.distances.append(unpacked_msg[index+1])
    
    # Print the robot sensor signal contents.
    def print(self):
        print("Robot Sensor Signal")
        print(" encoder: ", self.encoder_counts)
        print(" steering:" , self.steering)
        print(" num_lidar_rays: ", self.num_lidar_rays)
        print(" angles: ",self.angles)
        print(" distances: ", self.distances)
    
    # Convert the sensor signal to a list of ints and floats.
    def to_list(self):
        sensor_data_list = []
        sensor_data_list.append(self.encoder_counts)
        sensor_data_list.append(self.steering)
        sensor_data_list.append(self.num_lidar_rays)
        for i in range(self.num_lidar_rays):
            sensor_data_list.append(self.angles[i])
            sensor_data_list.append(self.distances[i])
        
        return sensor_data_list
