# External Libraries
import math
import random
from numpy.polynomial import Polynomial

import numpy as np

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of encoder counts
def variance_distance_travelled_s(encoder_counts):
    # From data fitting results (converted from Polynomial.fit scaled domain)
    p = Polynomial([-1.98084649e-04, 6.27582048e-08, -3.39443762e-12])
    var_s = max(p(encoder_counts), 1e-8)

    return var_s

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    # From data fitting results
    p = Polynomial([ 0.00000000e+00,  2.77722956e-04, -2.88552063e-11])
    s = p(encoder_counts)
    return s

# A function for obtaining variance in rotational velocity as a function of steering angle
# Piecewise degree-2 through origin: var_w = v1*a + v2*a^2
def variance_rotational_velocity_w(steering_angle):
    a = steering_angle
    if a < 0:
        var_w = -4.8725465517e-08 * a + -1.7897854046e-09 * a**2
    elif a > 0:
        var_w = 1.8038574058e-07 * a + -7.9043028987e-09 * a**2
    else:
        var_w = 1e-10
    return max(var_w, 1e-10)

def rotational_velocity_w(steering_angle_command):
    # Piecewise degree-2 through origin: w = c1*a + c2*a^2 (deg/ms)
    a = steering_angle_command
    if a < 0:
        w = 1.5607240143e-03 * a + 1.5412186380e-06 * a**2
    elif a > 0:
        w = 7.6584229391e-04 * a + 2.2293906810e-05 * a**2
    else:
        w = 0.0
    return w

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state, last_encoder_count):
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)
    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # Add student code here
        s = distance_travelled_s(encoder_counts - self.last_encoder_count)
        a = steering_angle_command
        w = rotational_velocity_w(a)
        var_s = variance_distance_travelled_s(encoder_counts - self.last_encoder_count)
        var_w = variance_rotational_velocity_w(a)   
        
        s += random.gauss(0, (var_s))
        w += random.gauss(0, (var_w))

        # w is in deg/ms from data fitting; convert to rad/s
        w_rad_s = w * 1000.0 * (math.pi / 180.0)

        delta_theta = w_rad_s * delta_t
        theta_mid = self.state[2] + delta_theta/2

        # Bicycle model 
        state = np.array(self.state) + np.array([
            s * math.cos(theta_mid),
            s * math.sin(theta_mid),
            delta_theta
        ])

        self.state = state.tolist()
        self.last_encoder_count = encoder_counts

        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list
    

    # Coming soon
    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t_list = [0]
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        t = 0
        encoder_counts = self.last_encoder_count
        while t < duration:
            #  a = theta / delta t
            steering_angle_command = -5
            encoder_counts += 100  # arbitrary increment for simulation
            new_state = self.step_update(encoder_counts, steering_angle_command, delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])
            t += delta_t
            t_list.append(t)
        return t_list, x_list, y_list, theta_list
            