# External libraries
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from sklearn.cluster import KMeans

# Local libraries
import parameters
import data_handling
from motion_models import *

# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

# Helper class to store and manipulate your states.
class State:

    # Constructor
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    # Get the euclidean distance between 2 states
    def distance_to(self, other_state):
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))
        
    # Get the distance squared between two states
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    # return a deep copy of the state.
    def deepcopy(self):
        return copy.deepcopy(self)
        
    # Print the state
    def print(self):
        print("State: ",self.x, self.y, self.theta)


# Class to store walls as objects (specifically when represented as line segments in a 2D map.)
class Wall:

    # Constructor
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)
        
        self.m = (wall_corners[3] - wall_corners[1])/(0.0001 + wall_corners[2] -  wall_corners[0])
        self.b = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm =  wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
        self.length = self.corner1.distance_to(self.corner2)
        self.length_mm_squared = self.corner1_mm.distance_to_squared(self.corner2_mm)
        
        if self.m > 1000:
            self.vertical = True
        else:
            self.vertical = False
        if abs(self.m) < 0.1:
            self.horizontal = True
        else:
            self.horizontal = False


# A class to store 2D maps
class Map:
    def __init__(self, wall_corner_list):
        self.wall_list = []
        for wall_corners in wall_corner_list:
            self.wall_list.append(Wall(wall_corners))
        min_x = 999999
        max_x = -99999
        min_y = 999999
        max_y = -99999
        for wall in self.wall_list:
            min_x = min(min_x, min(wall.corner1.x, wall.corner2.x))
            max_x = max(max_x, max(wall.corner1.x, wall.corner2.x))
            min_y = min(min_y, min(wall.corner1.y, wall.corner2.y))
            max_y = max(max_y, max(wall.corner1.y, wall.corner2.y))
        border = 0.5
        self.plot_range = [min_x - border, max_x + border, min_y - border, max_y + border]
        
        self.particle_range = [min_x , max_x , min_y, max_y]

        # Precompute wall geometry as numpy arrays for vectorized ray casting
        self._wall_A  = np.array([[w.corner1.x, w.corner1.y] for w in self.wall_list])  # (N, 2)
        self._wall_AB = np.array([[w.corner2.x - w.corner1.x,
                                   w.corner2.y - w.corner1.y] for w in self.wall_list])  # (N, 2)

    # Function to calculate the ray-cast distance from a state to the closest wall
    # in the direction of state.theta, using vectorized ray-segment intersection.
    def closest_distance_to_walls(self, state):
        d  = np.array([math.cos(state.theta), math.sin(state.theta)])
        P  = np.array([state.x, state.y])
        AP = P - self._wall_A                                             # (N, 2)
        # d × AB per wall: det[n] = d_x*AB_y[n] - d_y*AB_x[n]
        det  = d[0] * self._wall_AB[:, 1] - d[1] * self._wall_AB[:, 0]   # (N,)
        safe = np.abs(det) > 1e-10
        sd   = np.where(safe, det, 1.0)
        # t = -(AP × AB) / det  (distance along the ray)
        t = -(AP[:, 0] * self._wall_AB[:, 1] - AP[:, 1] * self._wall_AB[:, 0]) / sd
        # u = (AP × d) / det  (position along wall segment, must be in [0,1])
        u = (AP[:, 1] * d[0] - AP[:, 0] * d[1]) / sd
        mask = safe & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
        return float(np.min(t[mask])) if mask.any() else 999999999999.0


# Class to hold a particle
class Particle:
    
    def __init__(self):
        self.state = State(0, 0, 0)
        self.weight = 1
        
    # Function to create a new random particle state within a range
    def randomize_uniformly(self, xy_range):
        ################## Add student code here ###################
        x_min, x_max, y_min, y_max = xy_range
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        theta = np.random.uniform(-np.pi, np.pi)
        self.state = State(x, y, theta)
        self.weight = 1

    # Function to create a new random particle state with a normal distribution
    def randomize_around_initial_state(self, initial_state, state_stdev):
        ################## Add student code here ###################
        x = np.random.normal(initial_state.x, state_stdev.x)
        y = np.random.normal(initial_state.y, state_stdev.y)
        theta = np.random.normal(initial_state.theta, state_stdev.theta)
        self.state = State(x, y, theta)
        self.weight = 1
        
    # Function to take a particle and "randomly" propagate it forward according to a motion model.
    def propagate_state(self, last_state, delta_encoder_counts, steering, delta_t):
        ################## Add student code here ###################
        if abs(delta_encoder_counts) < 1:
            return  # Robot stationary — bicycle model has no rotation without forward motion

        s = distance_travelled_s(delta_encoder_counts)
        var_s = variance_distance_travelled_s(delta_encoder_counts)
        s += random.gauss(0, math.sqrt(max(var_s, 0)))

        w = rotational_velocity_w(steering)
        var_w = variance_rotational_velocity_w(steering)
        w += random.gauss(0, math.sqrt(max(var_w, 0)))

        x_tm1, y_tm1, theta_tm1 = last_state

        # w is in deg/ms from data fitting; convert to rad/s
        # Negate: camera world frame has opposite rotation handedness from compass convention
        w_rad_s = -w * 1000.0 * (math.pi / 180.0)
        delta_theta = w_rad_s * delta_t
        # Extra heading noise to let the PF compensate for rotation model inaccuracy
        delta_theta += random.gauss(0, 0.03)

        theta_mid = theta_tm1 + delta_theta / 2

        state = np.array(last_state) + np.array([
            s * math.cos(theta_mid),
            s * math.sin(theta_mid),
            delta_theta
        ]).tolist()

        self.state = State(*state)
        
        
    # Function to determine a particles weight based how well the lidar measurement matches up with the map.
    def calculate_weight(self, lidar_signal, map):
        ################## Add student code here ###################
        if not lidar_signal.angles:
            return

        # Subsample rays for speed, then convert units in one numpy pass
        step = parameters.lidar_ray_step
        angles_raw  = np.array(lidar_signal.angles[::step],    dtype=float)
        distances_m = np.array(lidar_signal.distances[::step], dtype=float) / 1000.0

        # Filter out-of-range readings (RPLidar returns large values when no surface detected)
        MAX_RANGE_M = 6.0
        valid_range = distances_m < MAX_RANGE_M
        angles_raw  = angles_raw[valid_range]
        distances_m = distances_m[valid_range]
        if len(distances_m) == 0:
            return

        # Hardware angle -> global frame: negate (CW->CCW) + deg->rad + robot heading
        global_angles = self.state.theta - angles_raw * (math.pi / 180.0)
        d_x = np.cos(global_angles)  # (M,)
        d_y = np.sin(global_angles)  # (M,)

        P  = np.array([self.state.x, self.state.y])
        AP = P - map._wall_A    # (N, 2)  vector from each wall-start to particle
        AB = map._wall_AB       # (N, 2)

        # Vectorized ray-segment intersection over all M rays × N walls
        # det[m,n] = d_x[m]*AB_y[n] - d_y[m]*AB_x[n]
        det  = np.outer(d_x, AB[:, 1]) - np.outer(d_y, AB[:, 0])  # (M, N)
        safe = np.abs(det) > 1e-10
        sd   = np.where(safe, det, 1.0)

        # t[m,n]: ray distance to wall n for ray m  (t = -(AP×AB) / det)
        t_num = -(AP[:, 0] * AB[:, 1] - AP[:, 1] * AB[:, 0])      # (N,)
        t = t_num[np.newaxis, :] / sd                               # (M, N)

        # u[m,n]: fractional position along wall n  (u = (AP×d) / det)
        u_num = np.outer(d_x, AP[:, 1]) - np.outer(d_y, AP[:, 0]) # (M, N)
        u = u_num / sd                                               # (M, N)

        valid = safe & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
        expected_distances = np.min(np.where(valid, t, np.inf), axis=1)  # (M,)

        # Robust likelihood: obstacle model for short readings, capped Gaussian for the rest
        diff = distances_m - expected_distances
        gaussian = -(diff ** 2) / (2.0 * parameters.distance_variance)
        if parameters.use_obstacle_buffer:
            contributions = np.where(
                diff < -parameters.obstacle_buffer_threshold,  # obstacle blocks ray (measured << expected)
                parameters.obstacle_buffer_penalty,            # fixed penalty for blocked rays
                np.maximum(gaussian, -1.0)                     # capped Gaussian for matches + through-wall
            )
        else:
            contributions = np.maximum(gaussian, -1.0)
        log_w = float(np.sum(contributions))
        self.weight *= math.exp(max(log_w, -700.0))
        
        
    # Return the log of the normal distribution function output.
    def log_gaussian(self, expected_distance, distance):
        return -math.pow(expected_distance - distance, 2) / 2.0 / parameters.distance_variance

    # Return the normal distribution function output.
    def gaussian(self, expected_distance, distance):
        return math.exp(self.log_gaussian(expected_distance, distance))

    # Deep copy the particle
    def deepcopy(self):
        return copy.deepcopy(self)
        
    # Print the particle
    def print(self):
        print("Particle: ", self.state.x, self.state.y, self.state.theta, " w: ", self.weight)


# This class holds the collection of particles.
class ParticleSet:
    
    # Constructor, which calls the known start or unknown start initialization.
    def __init__(self, num_particles, xy_range, initial_state, state_stdev, known_start_state):
        self.num_particles = num_particles
        self.particle_list = []
        self.xy_range = xy_range
        if known_start_state:
            print("Known")
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state = State(0, 0, 0)
        self._low_weight_streak = 0
        self.update_mean_state()
        
    # Function to reset particles on a uniform grid over the workspace.
    def generate_uniform_random_particles(self, xy_range):
        x_min, x_max, y_min, y_max = xy_range
        dx = x_max - x_min
        dy = y_max - y_min
        # Fix nth = cube-root of total, then scale nx/ny by aspect ratio
        # so spatial density (particles/m²) is equal in x and y.
        nth = max(2, int(round(self.num_particles ** (1.0 / 3.0))))
        nxy = self.num_particles / nth
        aspect = dx / dy
        nx = max(1, int(round(math.sqrt(nxy * aspect))))
        ny = max(1, int(round(math.sqrt(nxy / aspect))))
        xs = np.linspace(x_min, x_max, nx, endpoint=False) + dx / (2 * nx)
        ys = np.linspace(y_min, y_max, ny, endpoint=False) + dy / (2 * ny)
        ths = np.linspace(-math.pi, math.pi, nth, endpoint=False)
        self.particle_list = []
        for x in xs:
            for y in ys:
                for th in ths:
                    p = Particle()
                    p.state = State(x, y, th)
                    self.particle_list.append(p)
        self.num_particles = len(self.particle_list)

    # Function to reset particles, normally distributed around the initial state. 
    def generate_initial_state_particles(self, initial_state, state_stdev):
        for i in range(self.num_particles):
            random_particle = Particle()
            random_particle.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(random_particle)

    # Function to resample the particles set, i.e. make a new one with more copies of particles with higher weights.  
    def resample(self, max_weight):
        ################## Add student code here ###################
        # max_weight is unused
        
        # Normalize weights
        total = sum(p.weight for p in self.particle_list)

        # If all weights underflowed to zero, fall back to uniform resampling
        if total == 0.0:
            total = 1.0
            for p in self.particle_list:
                p.weight = 1.0

        # Reinitialize only after several consecutive cycles of very low weight
        max_w = max(p.weight for p in self.particle_list)
        self._last_pre_resample_max_w = max_w
        if max_w < parameters.reinit_weight_threshold:
            self._low_weight_streak += 1
        else:
            self._low_weight_streak = 0

        if self._low_weight_streak >= parameters.reinit_cycles_required:
            self._low_weight_streak = 0
            self.generate_uniform_random_particles(self.xy_range)
            return

        # Resampling and reassigning weight = 1 | O(N log N)
        new_particle_list = []
        for p in random.choices(
            self.particle_list,
            weights=[p.weight / total for p in self.particle_list],
            k=self.num_particles
        ):
            p = p.deepcopy()
            p.weight = 1
            new_particle_list.append(p)
        self.particle_list = new_particle_list


    # Calculate the mean state using K-means to find the dominant cluster.
    def update_mean_state(self):
        # Build (N, 3) feature matrix: x, y, and angle encoded as (sin, cos) projected to a single
        # value isn't enough — use (x, y, sin(θ), cos(θ)) but we need a scalar for KMeans.
        # Instead cluster on (x, y) and compute circular-mean θ from the winning cluster.
        pts = np.array([[p.state.x, p.state.y] for p in self.particle_list])
        weights = np.array([p.weight for p in self.particle_list], dtype=float)
        total_weight = weights.sum()
        if total_weight == 0:
            total_weight = 1.0
            weights = np.ones(len(self.particle_list))
        weights /= total_weight

        n_clusters = min(3, len(np.unique(pts, axis=0)))
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
        labels = kmeans.fit_predict(pts)

        # Pick the cluster whose members carry the most total weight
        cluster_weights = np.array([
            weights[labels == k].sum() for k in range(n_clusters)
        ])
        best = int(np.argmax(cluster_weights))
        mask = labels == best
        cluster_w = weights[mask]
        cluster_w /= cluster_w.sum()

        self.mean_state.x = float(np.dot(cluster_w, pts[mask, 0]))
        self.mean_state.y = float(np.dot(cluster_w, pts[mask, 1]))

        # Circular mean of θ within the winning cluster
        thetas = np.array([p.state.theta for p, m in zip(self.particle_list, mask) if m])
        self.mean_state.theta = math.atan2(
            float(np.dot(cluster_w, np.sin(thetas))),
            float(np.dot(cluster_w, np.cos(thetas))))
        
    # Print the particle set. Useful for debugging.
    def print_particles(self):
        for particle in self.particle_list:
            particle.print()
        print()

# Class to hold the particle filter and its functions.
class ParticleFilter:
    
    # Constructor
    def __init__(self, num_particles, map, initial_state, state_stdev, known_start_state, encoder_counts_0):
        self.map = map
        self.particle_set = ParticleSet(num_particles, map.particle_range, initial_state, state_stdev, known_start_state)
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list = []
        self.last_time = 0
        self.last_encoder_counts = encoder_counts_0

    # Update the states given new measurements
    def update(self, odometery_signal, measurement_signal, delta_t):
        self.prediction(odometery_signal, delta_t)
        if len(measurement_signal.angles)>0:
            self.correction(measurement_signal)
        self.particle_set.update_mean_state()
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    # Predict the current state from the last state.
    def prediction(self, odometry_signal, delta_t):
        ################## Add student code here ###################
        # Calculate the change in encoder counts from the last time step. Leverage self.last_encoder counts
        # odometry_signal has two elements, encoder_counts and steering angle
        # Next use a motion model to randomly propagate all particles from a deep copy of their current state. 
        # Be sure to use the Particle class propagate state function.

        encoder_count, steering = odometry_signal
        delta_encoder_counts = encoder_count - self.last_encoder_counts


        for particle in self.particle_set.particle_list:
            particle.propagate_state(
                (particle.state.x, particle.state.y, particle.state.theta), 
                delta_encoder_counts, 
                steering, 
                delta_t)
            
        self.last_encoder_counts = encoder_count

        return
        
    # Corrrect the predicted states.
    def correction(self, measurement_signal):
        ################## Add student code here ###################
        # Determine the max weight and use it to resample the particle set.
        # max_weight is unused
        max_weight = 0
        
        # CAlculate weights based on measurement
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)

        self.particle_set.resample(max_weight)

    # Output to terminal the mean state.
    def print_state_estimate(self):
        print("Mean state: ", self.particle_set.mean_state.x, self.particle_set.mean_state.y, self.particle_set.mean_state.theta)
    

# Class to help with plotting PF data.
class ParticleFilterPlot:

    # Constructor
    def __init__(self, map):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        self.map = map

    # Clear and update the plot with new PF data
    def update(self, state_mean, particle_set, lidar_signal, hold_show_plot):
        plt.clf()
        
        # Plot walls
        for wall in self.map.wall_list:
            plt.plot([wall.corner1.x, wall.corner2.x],[wall.corner1.y, wall.corner2.y],'k')

        # Plot lidar
        for i in range(len(lidar_signal.angles)):
            distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta
            x_ray = [state_mean.x, state_mean.x + distance * math.cos(angle)]
            y_ray = [state_mean.y, state_mean.y + distance * math.sin(angle)]
            plt.plot(x_ray, y_ray, 'r')


        # Plot state estimate
        plt.plot(state_mean.x, state_mean.y,'ro')
        plt.plot(
            [state_mean.x, state_mean.x+ self.dir_length*math.cos(state_mean.theta) ],
            [state_mean.y, state_mean.y+ self.dir_length*math.sin(state_mean.theta) ],'r')
        x_particles, y_particles = self.to_plot_data(particle_set)
        plt.plot(x_particles, y_particles, 'g.')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis(self.map.plot_range)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.grid()
        if hold_show_plot:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)

    # Helper function to make the particles easy to plot.
    def to_plot_data(self, particle_set):
        x_list = []
        y_list = []
        for p in particle_set.particle_list:
            x_list.append(p.state.x)
            y_list.append(p.state.y)
        return x_list, y_list
        

def find_initial_pose(map_obj, lidar_signal):
    """Grid-search the first LiDAR scan to find the best (x, y, theta)."""
    step = parameters.lidar_ray_step
    angles_raw  = np.array(lidar_signal.angles[::step],    dtype=float)
    distances_m = np.array(lidar_signal.distances[::step], dtype=float) / 1000.0

    # Filter out-of-range readings
    MAX_RANGE_M = 6.0
    valid_range = distances_m < MAX_RANGE_M
    angles_raw  = angles_raw[valid_range]
    distances_m = distances_m[valid_range]

    AB     = map_obj._wall_AB       # (N, 2)
    wall_A = map_obj._wall_A        # (N, 2)

    x_lo, x_hi, y_lo, y_hi = map_obj.particle_range
    best_lw = -np.inf
    best_pose = (0.0, 0.0, 0.0)

    # --- coarse pass (0.2 m, 10°) ---
    xs  = np.arange(x_lo, x_hi + 0.01, 0.2)
    ys  = np.arange(y_lo, y_hi + 0.01, 0.2)
    ths = np.arange(-math.pi, math.pi, math.radians(10))

    for x in xs:
        for y in ys:
            P    = np.array([x, y])
            AP   = P - wall_A                                              # (N, 2)
            t_num = -(AP[:, 0] * AB[:, 1] - AP[:, 1] * AB[:, 0])         # (N,)
            for th in ths:
                ga  = th - angles_raw * (math.pi / 180.0)
                dx  = np.cos(ga)
                dy  = np.sin(ga)
                det = np.outer(dx, AB[:, 1]) - np.outer(dy, AB[:, 0])     # (M, N)
                safe = np.abs(det) > 1e-10
                sd   = np.where(safe, det, 1.0)
                t    = t_num[np.newaxis, :] / sd
                u    = (np.outer(dx, AP[:, 1]) - np.outer(dy, AP[:, 0])) / sd
                valid = safe & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
                exp_d = np.min(np.where(valid, t, np.inf), axis=1)
                diff  = distances_m - exp_d
                gauss = -(diff ** 2) / (2.0 * parameters.distance_variance)
                contrib = np.where(diff < -0.3, -0.3, np.maximum(gauss, -1.0))
                lw = float(np.sum(contrib))
                if lw > best_lw:
                    best_lw   = lw
                    best_pose = (x, y, th)

    # --- fine refinement (0.05 m, 2°) around the coarse best ---
    bx, by, bth = best_pose
    for x in np.arange(bx - 0.2, bx + 0.201, 0.05):
        for y in np.arange(by - 0.2, by + 0.201, 0.05):
            P    = np.array([x, y])
            AP   = P - wall_A
            t_num = -(AP[:, 0] * AB[:, 1] - AP[:, 1] * AB[:, 0])
            for th in np.arange(bth - math.radians(10), bth + math.radians(10.01), math.radians(2)):
                ga  = th - angles_raw * (math.pi / 180.0)
                dx  = np.cos(ga)
                dy  = np.sin(ga)
                det = np.outer(dx, AB[:, 1]) - np.outer(dy, AB[:, 0])
                safe = np.abs(det) > 1e-10
                sd   = np.where(safe, det, 1.0)
                t    = t_num[np.newaxis, :] / sd
                u    = (np.outer(dx, AP[:, 1]) - np.outer(dy, AP[:, 0])) / sd
                valid = safe & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
                exp_d = np.min(np.where(valid, t, np.inf), axis=1)
                diff  = distances_m - exp_d
                gauss = -(diff ** 2) / (2.0 * parameters.distance_variance)
                contrib = np.where(diff < -0.3, -0.3, np.maximum(gauss, -1.0))
                lw = float(np.sum(contrib))
                if lw > best_lw:
                    best_lw   = lw
                    best_pose = (x, y, th)

    print(f"Pre-localization: ({best_pose[0]:.3f}, {best_pose[1]:.3f}, "
          f"{math.degrees(best_pose[2]):.1f}°)  log_w={best_lw:.2f}")
    return State(best_pose[0], best_pose[1], best_pose[2])


# Function used to test your PF offline with logged data.
def offline_pf():
    
    # Make a map of walls
    map = Map(parameters.wall_corner_list)

    # Get data to filter
    # filename = './data/robot_data_0_0_25_02_26_21_41_33.pkl'

    filename = './data/robot_data_73_-20_13_03_26_19_01_49.pkl'
    pf_data = data_handling.get_file_data_for_pf(filename)

    # Automatic initial-pose estimation from first LiDAR scan
    # initial_pose = find_initial_pose(map, pf_data[0][2])
    initial_pose = State(2, 2, 90)
    particle_filter = ParticleFilter(
        parameters.num_particles,
        map, 
        initial_state=initial_pose, 
        state_stdev=State(0.3, 0.3, 0.3), 
        known_start_state=True, 
        encoder_counts_0=pf_data[0][2].encoder_counts
    )

    # Create plotting tool for particles
    particle_filter_plot = ParticleFilterPlot(map)

    # Plot initial uniform grid before any updates
    particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, pf_data[0][2], False)

    # Loop over pf data
    for t in range(1, len(pf_data)):
        row = pf_data[t]
        delta_t = pf_data[t][0] - pf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = row[2] # lidar_sensor_signal

        # Run the PF for a time step
        particle_filter.update(u_t, z_t, delta_t)
        particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

    particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

        


####### MAIN #######
if __name__ == '__main__':
    offline_pf()

