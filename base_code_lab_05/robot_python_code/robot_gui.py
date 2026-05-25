# External libraries
import asyncio
import cv2
import math
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from nicegui import ui, app, run
import numpy as np
from fastapi import Response
from time import time

# Local libraries
from robot import Robot
import robot_python_code
import parameters

stream_video = False


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


def get_time_in_ms():
    return int(time() * 1000)


@ui.page('/')
def main():

    robot = Robot()

    # LiDAR state in robot frame (one distance bucket per angular bin, metres)
    max_lidar_range = 12
    lidar_angle_res = 2
    num_angles = int(360 / lidar_angle_res)
    lidar_distance_list = [max_lidar_range] * num_angles

    dark = ui.dark_mode()
    dark.value = True

    if stream_video:
        video_capture = cv2.VideoCapture(parameters.camera_id)

    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return Response(content=b'', media_type='image/jpeg')
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return Response(content=b'', media_type='image/jpeg')
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    # ── Sensor helpers ────────────────────────────────────────────────

    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = robot.robot_sensor_signal.angles[i]  # keep raw CW hardware angle
            if distance_in_mm > 20 and abs(angle) < 360:
                idx = max(0, min(num_angles - 1,
                                 int((angle - lidar_angle_res / 2) / lidar_angle_res)))
                lidar_distance_list[idx] = distance_in_mm / 1000

    # ── Command helpers ───────────────────────────────────────────────

    def update_commands():
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                robot.extra_logging = True
        if robot.extra_logging:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time + parameters.extra_trial_log_time:
                logging_switch.value = False
                robot.extra_logging = False
        # cmd_speed: -1 = forced stop (overrides MPPI auto-speed), 0 = slider at zero, >0 = slider value
        cmd_speed = slider_speed.value if speed_switch.value else -1
        cmd_steering = slider_steering.value if steering_switch.value else 0
        return cmd_speed, cmd_steering

    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, success = robot_python_code.create_udp_communication(
                    parameters.arduinoIP, parameters.localIP,
                    parameters.arduinoPort, parameters.localPort,
                    parameters.bufferSize)
                if success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True

    def on_set_goal():
        try:
            gx = float(goal_x_input.value)
            gy = float(goal_y_input.value)
        except (TypeError, ValueError):
            mppi_status_label.set_text('Invalid coordinates')
            return
        std = robot.pf_std
        if std > 0.5:
            mppi_status_label.set_text(
                f'PF not converged (σ={std:.2f} m) — drive to localise first')
            mppi_status_label.style('color: orange;')
            return
        robot.set_goal(gx, gy)
        mppi_status_label.set_text(f'Goal ({gx:.2f}, {gy:.2f}) set — MPPI active')
        mppi_status_label.style('color: #44cc44;')
        # Clear pending inputs so the yellow marker disappears and only the red star shows
        goal_x_input.set_value(None)
        goal_y_input.set_value(None)

    # ── Map plot ──────────────────────────────────────────────────────

    def show_map_plot():
        mean = robot.particle_filter.particle_set.mean_state
        pr = robot.particle_filter.map.plot_range  # [xmin, xmax, ymin, ymax]

        with map_plot:
            fig = map_plot.fig
            fig.patch.set_facecolor('#1a1a1a')
            plt.clf()
            ax = fig.add_subplot(111)
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#aaaaaa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')

            # Walls
            for w in parameters.wall_corner_list:
                ax.plot([w[0], w[2]], [w[1], w[3]],
                        color='white', linewidth=1.5, zorder=2)

            # Particles (subsampled to keep redraw fast)
            particles = robot.particle_filter.particle_set.particle_list
            step = max(1, len(particles) // 150)
            px = [p.state.x for p in particles[::step]]
            py = [p.state.y for p in particles[::step]]
            ax.scatter(px, py, s=3, c='#4488ff', alpha=0.35, zorder=3,
                       label='Particles')

            # LiDAR rays projected into world frame (every 4th ray for speed)
            rx, ry, rtheta = mean.x, mean.y, mean.theta
            for i in range(0, num_angles, 4):
                dist = lidar_distance_list[i]
                if dist < max_lidar_range:
                    # Convention from particle_filter.py: global_angle = theta - angle_rad
                    global_angle = rtheta - i * lidar_angle_res * math.pi / 180.0
                    ex = rx + dist * math.cos(global_angle)
                    ey = ry + dist * math.sin(global_angle)
                    ax.plot([rx, ex], [ry, ey],
                            color='#cc2222', linewidth=0.6, alpha=0.55, zorder=4)

            # Sampled MPPI trajectories (single LineCollection draw call — fast)
            trajs = robot.last_mppi_trajectories
            if trajs is not None:
                segs = [trajs[k, :, :2] for k in range(trajs.shape[0])]
                ax.add_collection(LineCollection(
                    segs, colors='#ffaa00', linewidths=0.4, alpha=0.2, zorder=4))

            # Shortest-path reference from MPPI
            ref = robot.mppi_controller.ref_path_xy
            if ref is not None and len(ref) > 1:
                ax.plot(ref[:, 0], ref[:, 1],
                        color='cyan', linewidth=1.5, linestyle='--',
                        zorder=5, label='Path')

            # Goal marker
            goal = robot.mppi_controller.goal_state
            ax.plot(goal[0], goal[1], marker='*', color='red',
                    markersize=14, zorder=6, label='Goal',
                    markeredgecolor='white', markeredgewidth=0.5)

            # Robot pose: filled dot + heading arrow
            arrow_len = 0.25
            ax.annotate(
                '', xy=(rx + arrow_len * math.cos(rtheta),
                        ry + arrow_len * math.sin(rtheta)),
                xytext=(rx, ry),
                arrowprops=dict(arrowstyle='->', color='#ff44ff', lw=2),
                zorder=7)
            ax.plot(rx, ry, 'o', color='#ff44ff', markersize=7,
                    zorder=7, label='Robot')

            ax.set_xlim(pr[0], pr[1])
            ax.set_ylim(pr[2], pr[3])
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)', color='#aaaaaa')
            ax.set_ylabel('y (m)', color='#aaaaaa')
            ax.set_title('Map & Localisation', color='#cccccc', pad=6)
            ax.legend(facecolor='#2a2a2a', edgecolor='#555555',
                      labelcolor='white', fontsize=8, loc='upper right',
                      markerscale=1.2)
            fig.tight_layout()

    # ── Layout ────────────────────────────────────────────────────────

    with ui.card().classes('w-full items-center'):
        ui.label('ROB-GY 6213: Robot Navigation & Localization') \
            .style('font-size: 22px; font-weight: bold;')

    with ui.card().classes('w-full'):
        with ui.grid(columns=2).classes('w-full items-start'):

            # Left column — full map
            with ui.card().classes('w-full items-center'):
                map_plot = ui.pyplot(figsize=(7, 7))

            # Right column — status & controls
            with ui.card().classes('w-full gap-2'):

                ui.label('Status').style('font-size: 15px; font-weight: bold;')
                encoder_count_label = ui.label('Encoder: 0')
                pf_status_label = ui.label('PF: Localising…') \
                    .style('color: orange;')
                udp_status_label = ui.label('UDP: disconnected') \
                    .style('color: gray;')

                ui.separator()

                ui.label('Goal').style('font-size: 15px; font-weight: bold;')
                with ui.row().classes('items-end gap-2'):
                    goal_x_input = ui.number(
                        'X (m)', value=parameters.goal_state[0],
                        format='%.2f').classes('w-24')
                    goal_y_input = ui.number(
                        'Y (m)', value=parameters.goal_state[1],
                        format='%.2f').classes('w-24')
                    ui.button('Set Goal', on_click=on_set_goal)
                mppi_status_label = ui.label('MPPI: not started') \
                    .style('color: gray;')

                ui.separator()

                ui.label('Control').style('font-size: 15px; font-weight: bold;')
                logging_switch = ui.switch('Data Logging')
                udp_switch = ui.switch('Robot Connect')
                ui.button('Run Trial', on_click=run_trial)

    # Speed
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('SPEED:')
            with ui.card().classes('w-full items-center'):
                slider_speed = ui.slider(min=0, max=100, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_speed, 'value')
            with ui.card().classes('w-full items-center'):
                speed_switch = ui.switch('Enable')

    # Steering
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('STEER:')
            with ui.card().classes('w-full items-center'):
                slider_steering = ui.slider(min=-20, max=20, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_steering, 'value')
            with ui.card().classes('w-full items-center'):
                steering_switch = ui.switch('Enable')

    # ── Control loop (10 Hz) ──────────────────────────────────────────

    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering = update_commands()
        robot.control_loop(cmd_speed, cmd_steering, logging_switch.value)
        encoder_count_label.set_text(
            f'Encoder: {robot.robot_sensor_signal.encoder_counts}')
        update_lidar_data()

        # UDP / send status
        if robot.connected_to_hardware and robot.msg_sender is not None:
            # Show the last command actually sent (clamp -1 sentinel for display)
            spd = max(0, cmd_speed)
            udp_status_label.set_text(f'UDP: connected  |  cmd speed={spd}  steer={cmd_steering:.1f}')
            udp_status_label.style('color: #44cc44;')
        else:
            udp_status_label.set_text('UDP: disconnected')
            udp_status_label.style('color: gray;')

        # PF convergence indicator
        std = robot.pf_std
        if std < 0.3:
            pf_status_label.set_text(f'PF: Converged  (σ={std:.2f} m)')
            pf_status_label.style('color: #44cc44;')
        else:
            pf_status_label.set_text(f'PF: Localising…  (σ={std:.2f} m)')
            pf_status_label.style('color: orange;')

        # MPPI goal-reached indicator (robot.py deactivates mppi_active on arrival)
        if not robot.mppi_active and mppi_status_label.text.startswith('Goal'):
            mppi_status_label.set_text('Goal reached — stopped')
            mppi_status_label.style('color: #44cc44;')

    ui.timer(0.1, control_loop)
    ui.timer(0.5, show_map_plot)  # map redraw at 2 Hz — keeps control loop fast


# Run the gui
ui.run(native=True)

