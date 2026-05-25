import socket
import csv
import time
import sys

ROBOT_IP  = "10.0.0.99"   # REPLACE with robot's IP shown in Serial Monitor after boot
LOCAL_IP  = "10.0.0.10"    # listen on all interfaces
PORT      = 4010
DURATION  = 10            # seconds to collect by default
OUTPUT    = "lidar_map.csv"

def main():
    duration   = int(sys.argv[1])   if len(sys.argv) > 1 else DURATION
    robot_ip   = sys.argv[2]        if len(sys.argv) > 2 else ROBOT_IP
    output_file = sys.argv[3]       if len(sys.argv) > 3 else OUTPUT

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LOCAL_IP, PORT))
    sock.settimeout(0.5)

    # Retry sending 'start' until the first packet arrives (robot may still be booting/connecting)
    print(f"Sending 'start' to {robot_ip}:{PORT} — waiting for first packet...")
    first_packet = False
    retry_interval = 2.0   # re-send 'start' every 2 s until robot responds
    last_start_send = time.time()
    wait_deadline = time.time() + 30   # give up waiting after 30 s

    sock.sendto(b"start", (robot_ip, PORT))

    while not first_packet:
        if time.time() > wait_deadline:
            print("ERROR: No data received from robot after 30 s. "
                  "Check that the robot is powered on, connected to WiFi, "
                  "and that ROBOT_IP / LOCAL_IP are correct.")
            sock.close()
            return
        if time.time() - last_start_send >= retry_interval:
            print(f"  No response yet — re-sending 'start' to {robot_ip}:{PORT}...")
            sock.sendto(b"start", (robot_ip, PORT))
            last_start_send = time.time()
        try:
            data, addr = sock.recvfrom(4096)
            print(f"  First packet received from {addr}!")
            first_packet = True
            # Parse and store this first packet below (fall through)
        except socket.timeout:
            continue

    readings = []
    # Parse the first packet that broke us out of the wait loop
    try:
        parts = data.decode().strip().split(',')
        num_rays = int(parts[0])
        for i in range(num_rays):
            angle = int(parts[1 + i * 2])
            dist  = int(parts[2 + i * 2])
            readings.append((angle, dist))
    except (ValueError, IndexError) as e:
        print(f"  Warning: could not parse first packet: {e}")

    start_time = time.time()
    last_report = start_time
    rays_since_report = len(readings)
    print(f"Collecting for {duration} seconds...")

    while time.time() - start_time < duration:
        # Progress report every second
        now = time.time()
        if now - last_report >= 1.0:
            elapsed = now - start_time
            rate = rays_since_report / (now - last_report)
            print(f"  {elapsed:.0f}s elapsed — {len(readings)} rays total, ~{rate:.0f} rays/s")
            if rays_since_report == 0:
                print("  WARNING: No new rays received in the last second. "
                      "LiDAR may not be spinning or robot may have stopped responding.")
            rays_since_report = 0
            last_report = now

        try:
            data, _ = sock.recvfrom(4096)
            parts = data.decode().strip().split(',')
            num_rays = int(parts[0])
            for i in range(num_rays):
                angle = int(parts[1 + i * 2])
                dist  = int(parts[2 + i * 2])
                readings.append((angle, dist))
            rays_since_report += num_rays
        except socket.timeout:
            pass
        except (ValueError, IndexError) as e:
            print(f"  Warning: malformed packet ignored ({e})")

    print(f"Sending 'stop' to {robot_ip}:{PORT}")
    sock.sendto(b"stop", (robot_ip, PORT))
    sock.close()

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(readings)

    print(f"Saved {len(readings)} readings to '{output_file}'")

if __name__ == "__main__":
    main()
