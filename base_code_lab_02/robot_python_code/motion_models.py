# External Libraries
import math
import random
from numpy.polynomial import Polynomial

import numpy as np

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance):
    # Add student code here
    # From data fitting results
    p = Polynomial([ 9.18891980e-05,  5.50499436e-06, -7.34909627e-05])
    var_s = p(distance)

    return var_s

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    # From data fitting results
    p = Polynomial([ 0.00000000e+00,  2.77722956e-04, -2.88552063e-11])
    s = p(encoder_counts)
    return s

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(distance):
    # Add student code here
    # From data fitting results
    p = Polynomial([ 6.06057171e-06, -4.07857014e-07, -4.19866455e-06])
    var_w = p(distance)

    return var_w

def rotational_velocity_w(steering_angle_command):
    # Add student code here
    # From data fitting results
    p = Polynomial([ 0.00503049, -0.0039763,   0.02361841])
    w = p(steering_angle_command)
    
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
        var_s = variance_distance_travelled_s(s)
        var_w = variance_rotational_velocity_w(s)   
        
        s += random.gauss(0, (var_s))
        w += random.gauss(0, (var_w))


        theta = w * delta_t
        delta_theta = theta - self.state[2] 

        state = np.array(self.state) + np.array([
            s * math.cos(self.state[2] + delta_theta/2),
            s * math.sin(self.state[2] + delta_theta/2),
            theta
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
            