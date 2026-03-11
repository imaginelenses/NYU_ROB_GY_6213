import socket
import csv
import time
import sys

ROBOT_IP  = "10.0.0.97"   # REPLACE with robot's IP shown in Serial Monitor after boot
LOCAL_IP  = "0.0.0.0"    # listen on all interfaces
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

    print(f"Sending 'start' to {robot_ip}:{PORT}")
    sock.sendto(b"start", (robot_ip, PORT))

    readings = []
    start_time = time.time()
    print(f"Collecting for {duration} seconds...")

    while time.time() - start_time < duration:
        try:
            data, _ = sock.recvfrom(4096)
            parts = data.decode().strip().split(',')
            num_rays = int(parts[0])
            for i in range(num_rays):
                angle = int(parts[1 + i * 2])
                dist  = int(parts[2 + i * 2])
                readings.append((angle, dist))
        except socket.timeout:
            pass
        except (ValueError, IndexError):
            pass

    print(f"Sending 'stop' to {robot_ip}:{PORT}")
    sock.sendto(b"stop", (robot_ip, PORT))
    sock.close()

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(readings)

    print(f"Saved {len(readings)} readings to '{output_file}'")

if __name__ == "__main__":
    main()
