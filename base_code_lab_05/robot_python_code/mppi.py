import numpy as np
import matplotlib.pyplot as plt
from edt import EDT
from motion_models import *
import parameters

class MPPI:
    def __init__(self, goal_state, goal_weight=5, K=100, T=30, alpha=0.5,
                 delta_t=0.1, nominal_encoder_rate=89.6, path_weight=1.0,
                 sigma_steering=6.0, sigma_encoder=0.0, heading_weight=0.0,
                 wall_weight=5.0):
        self.goal_state = goal_state
        self.goal_weight = goal_weight
        self.path_weight = path_weight
        self.heading_weight = heading_weight
        self.wall_weight = wall_weight

        self.K = K
        self.T = T
        self.alpha = alpha

        self.delta_t = delta_t
        self.nominal_encoder_rate = nominal_encoder_rate

        self.edt = EDT(parameters.wall_corner_list)

        # Warm-start nominal control sequence, shape (T, 2)
        self.U_nominal = np.zeros((T, 2))
        self.U_nominal[:, 0] = self.nominal_encoder_rate

        # Exploration noise for trajectory sampling — independent of motion model process noise.
        # sigma_steering in servo degrees; sigma_encoder in encoder counts per step.
        self.sigma_steering = sigma_steering
        self.sigma_encoder  = sigma_encoder if sigma_encoder > 0.0 else np.sqrt(variance_distance_travelled_s(self.nominal_encoder_rate))

        self.set_goal(goal_state)


    def set_goal(self, goal_state, start_state=None):
        """Set or update the goal and recompute the reference shortest path.

        start_state: (x, y) or (x, y, theta); if None, path starts from goal_state
        (a full path will be computed once a real start is known via the first
        mppi_iteration call, but this pre-warms ref_path_xy to avoid None checks).
        """
        self.goal_state = goal_state
        if start_state is not None:
            start_x, start_y = start_state[0], start_state[1]
        else:
            start_x, start_y = goal_state[0], goal_state[1]
        ref_path = self.edt.shortest_path(start_x, start_y, goal_state[0], goal_state[1])
        self.ref_path_xy = np.array(ref_path) if ref_path is not None else None  # shape (P, 2)

    def _propagate_batch(self, initial_state, controls):
        """Vectorised noise-free trajectory propagation for all K trajectories at once.

        Uses the mean motion model (no process noise — control perturbations already
        provide diversity). ~50x faster than the Python loop version.

        initial_state: [x, y, theta]
        controls: (K, T, 2) — column 0 = encoder increment per step, col 1 = steering angle
        Returns trajectories (K, T+1, 3)
        """
        # Polynomial coefficients (from motion_models.py)
        # s = c1*e + c2*e^2
        _S1, _S2 = 2.77722956e-04, -2.88552063e-11
        # w piecewise: neg: c1*a + c2*a^2 ; pos: c1*a + c2*a^2  (deg/ms)
        _WN1, _WN2 =  1.5607240143e-03,  1.5412186380e-06
        _WP1, _WP2 =  7.6584229391e-04,  2.2293906810e-05

        trajs = np.zeros((self.K, self.T + 1, 3))
        trajs[:, 0, :] = initial_state

        enc_delta = controls[:, :, 0]          # (K, T)
        steer     = controls[:, :, 1]          # (K, T)

        # Distance from encoder increment
        s = _S1 * enc_delta + _S2 * enc_delta ** 2  # (K, T)
        # Zero out nearly-stationary steps
        s[np.abs(enc_delta) < 1] = 0.0

        # Rotational velocity (piecewise)
        w = np.where(steer < 0,
                     _WN1 * steer + _WN2 * steer ** 2,
                     np.where(steer > 0,
                               _WP1 * steer + _WP2 * steer ** 2,
                               0.0))                     # (K, T)
        # Convert deg/ms → rad/s, negate (handedness)
        w_rad_s = -w * 1000.0 * (np.pi / 180.0)
        delta_theta = w_rad_s * self.delta_t             # (K, T)

        # Integrate step-by-step (theta depends on previous theta, must loop over T)
        for t in range(self.T):
            theta = trajs[:, t, 2]
            theta_mid = theta + delta_theta[:, t] / 2
            trajs[:, t + 1, 0] = trajs[:, t, 0] + s[:, t] * np.cos(theta_mid)
            trajs[:, t + 1, 1] = trajs[:, t, 1] + s[:, t] * np.sin(theta_mid)
            trajs[:, t + 1, 2] = theta + delta_theta[:, t]

        return trajs

    def sample_trajectories(self, initial_state, last_encoder_count):
        """Sample K trajectories of length T from the initial state."""
        encoder_epsilons = np.random.normal(0.0, self.sigma_encoder, (self.K, self.T))
        steering_epsilons = np.random.normal(0.0, self.sigma_steering, (self.K, self.T))

        controls = np.zeros((self.K, self.T, 2))
        controls[:, :, 0] = self.U_nominal[:, 0] + encoder_epsilons
        controls[:, :, 1] = np.clip(self.U_nominal[:, 1] + steering_epsilons, -20.0, 20.0)

        trajectories = self._propagate_batch(initial_state, controls)
        return trajectories, controls

    def compute_costs(self, trajectories):
        """Compute the cost of each trajectory — fully vectorised (no Python loops over K or T)."""
        costs = np.zeros(self.K)

        # ── Wall proximity cost (vectorised) ─────────────────────────────────
        # trajectories: (K, T+1, 3); use steps 0..T-1
        xs = trajectories[:, :self.T, 0]   # (K, T)
        ys = trajectories[:, :self.T, 1]   # (K, T)

        # Convert world coords to grid indices — vectorised
        i_idx = np.clip(
            ((xs - self.edt.x_min) / self.edt.cell_size).astype(int),
            0, self.edt.x_cells - 1)
        j_idx = np.clip(
            ((ys - self.edt.y_min) / self.edt.cell_size).astype(int),
            0, self.edt.y_cells - 1)

        edt_vals = self.edt.edt_grid[j_idx, i_idx]  # (K, T)
        in_wall = edt_vals <= 0.0
        costs += np.sum(in_wall * 1000.0
                        + (~in_wall) * self.wall_weight * np.exp(-edt_vals / 0.5),
                        axis=1)

        # ── Path-following + heading cost (vectorised) ───────────────────────
        if self.ref_path_xy is not None and len(self.ref_path_xy) > 1:
            # Nearest-point distance to reference path for every (k,t) — (K, T, P) broadcast
            dx = xs[:, :, None] - self.ref_path_xy[None, None, :, 0]   # (K,T,P)
            dy = ys[:, :, None] - self.ref_path_xy[None, None, :, 1]   # (K,T,P)
            dists = np.hypot(dx, dy)                                     # (K,T,P)
            nearest_idx = np.argmin(dists, axis=2)                       # (K,T)
            nearest_dist = dists[
                np.arange(self.K)[:, None],
                np.arange(self.T)[None, :],
                nearest_idx]                                              # (K,T)
            costs += self.path_weight * np.sum(nearest_dist, axis=1)

            if self.heading_weight > 0.0:
                next_idx = np.minimum(nearest_idx + 1, len(self.ref_path_xy) - 1)
                # Path tangent direction at nearest point
                dp = self.ref_path_xy[next_idx] - self.ref_path_xy[nearest_idx]  # (K,T,2)
                path_heading = np.arctan2(dp[:, :, 1], dp[:, :, 0])              # (K,T)
                robot_heading = trajectories[:, :self.T, 2]                       # (K,T)
                # Only penalise steps where next != nearest (not at path end)
                valid = next_idx > nearest_idx
                heading_err = np.abs(np.arctan2(
                    np.sin(path_heading - robot_heading),
                    np.cos(path_heading - robot_heading)))                         # (K,T)
                costs += self.heading_weight * np.sum(heading_err * valid, axis=1)

        # ── Terminal goal cost ───────────────────────────────────────────────
        x_T = trajectories[:, -1, 0]
        y_T = trajectories[:, -1, 1]
        costs += self.goal_weight * np.hypot(
            x_T - self.goal_state[0], y_T - self.goal_state[1])

        return costs

    def update_controls(self, controls, costs):
        """Update U_nominal via softmax weighting, then shift it forward by one step."""
        costs_shifted = costs - costs.min()
        weights = np.exp(-costs_shifted / self.alpha)
        weights /= np.sum(weights)

        # Weighted average over K trajectories -> new nominal sequence, shape (T, 2)
        self.U_nominal = np.sum(weights[:, None, None] * controls, axis=0)
        self.U_nominal[:, 1] = np.clip(self.U_nominal[:, 1], -20.0, 20.0)

        # Shift left by one timestep; pad the last step with zero steering, nominal speed
        self.U_nominal = np.roll(self.U_nominal, -1, axis=0)
        self.U_nominal[-1, 0] = self.nominal_encoder_rate
        self.U_nominal[-1, 1] = 0.0

        return self.U_nominal

    
    def visualize_trajectories(self, trajectories):
        """Visualize the sampled trajectories on top of the EDT."""
        self.edt.visualize()
        for k in range(self.K):
            plt.plot(trajectories[k, :, 0], trajectories[k, :, 1], alpha=0.3)
        plt.scatter(self.goal_state[0], self.goal_state[1], c='red', marker='*', s=200, label='Goal')
        plt.legend()
        plt.title('Sampled Trajectories on EDT')
        plt.show()

    def mppi_iteration(self, initial_state, last_encoder_count):
        trajectories, controls = self.sample_trajectories(initial_state, last_encoder_count)

        trajectories, controls = self.sample_trajectories(initial_state, last_encoder_count)
        costs = self.compute_costs(trajectories)
        new_U_nominal = self.update_controls(controls, costs)

        self.U_nominal = new_U_nominal

        return self.U_nominal, trajectories
    
