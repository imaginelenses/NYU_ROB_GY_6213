# External libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from numpy.polynomial import Polynomial

# Local libraries
import parameters
import data_handling
import motion_models

# Main class
class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = x_0
        self.state_covariance = Sigma_0
        self.predicted_state_mean = [0,0,0]
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = encoder_counts_0

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        # Check stationarity BEFORE prediction_step updates last_encoder_counts
        encoder_change = abs(u_t[0] - self.last_encoder_counts)

        if encoder_change < 1:
            # Robot stationary, keep current state unchanged (no rotation when not moving)
            self.last_encoder_counts = u_t[0]
            return

        self.prediction_step(u_t, delta_t)

        if z_t is not None:
            self.correction_step(z_t)
        else:
            # Moving but no measurement available, use prediction only
            self.state_mean = self.predicted_state_mean
            self.state_covariance = self.predicted_state_covariance
        return
    
    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        H_t = self.get_H()
        Q_t = self.get_Q()

        z_pred = self.get_h_function(self.predicted_state_mean)
        y_t = z_t - z_pred
        # Wrap theta residual to [-pi, pi] (camera uses arctan2, model accumulates continuously)
        y_t[2] = (y_t[2] + math.pi) % (2 * math.pi) - math.pi
        S_t = H_t @ self.predicted_state_covariance @ H_t.T + Q_t

        # # Mahalanobis distance
        # d_squared = y_t.T @ np.linalg.inv(S_t) @ y_t
        # if d_squared > 11.345:  # chi-square 99% threshold for 3 DOF
        #     # Reject outlier measurement
        #     self.state_mean = self.predicted_state_mean
        #     self.state_covariance = self.predicted_state_covariance
        #     return

        K_t = self.predicted_state_covariance @ H_t.T @ np.linalg.inv(S_t)

        self.state_mean = self.predicted_state_mean + K_t @ y_t
        self.state_covariance = (parameters.I3 - K_t @ H_t) @ self.predicted_state_covariance

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        encoder_counts, steering_angle = u_t
        encoder_change = encoder_counts - self.last_encoder_counts

        x_t, s = self.g_function(self.state_mean, u_t, delta_t)
        G_x = self.get_G_x(self.state_mean, u_t, delta_t)
        G_u = self.get_G_u(self.state_mean, u_t, delta_t)
        R_t = self.get_R(encoder_change, steering_angle, delta_t)

        self.predicted_state_mean = x_t
        self.predicted_state_covariance = G_x @ self.state_covariance @ G_x.T + G_u @ R_t @ G_u.T

        # Update encoder counts AFTER all computations that depend on the difference
        self.last_encoder_counts = u_t[0]
        return

    # The nonlinear transition equation that provides new states from past states
    # Motion model - Bicycle model
    def g_function(self, x_tm1, u_t, delta_t):
        encoder_counts, steering_angle = u_t
        s = motion_models.distance_travelled_s(encoder_counts - self.last_encoder_counts)
        w = motion_models.rotational_velocity_w(steering_angle)

        # w is in deg/ms from data fitting; convert to rad/s
        # Negate: camera world frame has opposite rotation handedness from compass convention
        w_rad_s = -w * 1000.0 * (math.pi / 180.0)

        delta_theta = w_rad_s * delta_t
        theta_tm1 = x_tm1[2]
        theta_mid = theta_tm1 + delta_theta / 2

        # Bicycle model 
        x_t = np.array(x_tm1) + np.array([
            s * math.cos(theta_mid),
            s * math.sin(theta_mid),
            delta_theta
        ])

        return x_t, s
    
    # The nonlinear measurement function
    def get_h_function(self, x_t):
        z_t = x_t
        return z_t
    
    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x_tm1, u_t, delta_t):
        x_t, s = self.g_function(x_tm1, u_t, delta_t)
        theta_t = x_t[2]
        theta_tm1 = x_tm1[2]

        delta_theta = theta_t - theta_tm1
        theta_mid = theta_tm1 + delta_theta/2
        
        G_x = np.array([
            [1, 0, -s * math.sin(theta_mid)],
            [0, 1,  s * math.cos(theta_mid)],
            [0, 0,  1]
        ])
        
        return G_x

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1, u_t, delta_t): 
        x_t, s = self.g_function(x_tm1, u_t, delta_t)
        theta_t = x_t[2]
        theta_tm1 = x_tm1[2]

        delta_theta = theta_t - theta_tm1
        theta_mid = theta_tm1 + delta_theta/2

        G_u = np.array([
            [math.cos(theta_mid), -s * math.sin(theta_mid) / 2],
            [math.sin(theta_mid),  s * math.cos(theta_mid) / 2],
            [0, 1]
        ])

        return G_u

    # This function returns a matrix with the partial derivatives dh_t/dx_t
    def get_H(self):
        return parameters.I3
    
    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, encoder_change, steering_angle, delta_t=0.1):
        var_s = motion_models.variance_distance_travelled_s(encoder_change)
        # Polynomial fitted for total-trial encoder counts [4417..13723];
        # per-step changes (~100) are outside valid domain → negative → clamped to 1e-8.
        # Use a physically meaningful floor: σ_s ≈ 1cm → var_s = 1e-4 m²
        var_s = max(var_s, 1e-4)

        # var_w is in (deg/ms)²; G_u expects var(delta_theta) in rad²
        # delta_theta = w * (1000 * π/180) * dt, so var_delta_theta = var_w * (1000*π/180)² * dt²
        var_w_deg_ms = motion_models.variance_rotational_velocity_w(steering_angle)
        var_delta_theta = var_w_deg_ms * (1000.0 * math.pi / 180.0) ** 2 * delta_t ** 2
        # Floor: σ_δθ ≈ 1° ≈ 0.017 rad → var = 3e-4 rad²
        var_delta_theta = max(var_delta_theta, 3e-4)
        R_t = np.array([
            [var_s, 0],
            [0, var_delta_theta]
        ])
        return R_t

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        # From ground truth experiment (camera_tracking.ipynb) in SI units (meters, radians)
        Q_t = np.array([[ 0.00169442,  0.00043347, -0.00131223],
                        [ 0.00043347,  0.00165269, -0.0010776 ],
                        [-0.00131223, -0.0010776 ,  0.0034934 ]])
        return Q_t

class KalmanFilterPlot:

    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

    def update(self, state_mean, state_covaraiance):
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covaraiance)
        lambda_ = np.sqrt(lambda_)
        # lambda_ *= 500 
        
        xy = (state_mean[0], state_mean[1])
        angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))

        ell = Ellipse(xy, alpha=0.5, facecolor='red', width=lambda_[0], height=lambda_[1], angle = angle)

        ax = self.fig.gca()
        ax.add_artist(ell)


        
        # Plot state estimate
        plt.plot(state_mean[0], state_mean[1],'ro')
        plt.plot([state_mean[0], state_mean[0]+ self.dir_length*math.cos(state_mean[2]) ], [state_mean[1], state_mean[1]+ self.dir_length*math.sin(state_mean[2]) ],'r')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        
        # plt.axis([-0.25, 2, -1, 1])
        plt.axis([-1, 3, -1, 1.5])

        plt.grid()
        plt.draw()
        plt.pause(0.1)


# Code to run your EKF offline with a data file.
def offline_efk():

    # Get data to filter
    # filename = './data/robot_data_0_-20_26_02_26_23_40_54.pkl'
    # filename = './data/robot_data_0_0_27_02_26_17_57_17.pkl'
    filename = './data/robot_data_0_0_27_02_26_18_30_11.pkl'

    ekf_data = data_handling.get_file_data_for_kf(filename)

    x_0 = [0, 0, 0]
    Sigma_0 = parameters.I3
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()

    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = row[3] # camera_sensor_signal

        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t)
        kalman_filter_plot.update(extended_kalman_filter.state_mean, extended_kalman_filter.state_covariance[0:2,0:2])


####### MAIN #######
if __name__ == "__main__":
    offline_efk()
