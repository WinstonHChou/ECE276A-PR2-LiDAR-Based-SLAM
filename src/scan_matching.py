from pr2_utils import *

if __name__ == '__main__':
  
  args = parse_args(sys.argv)
  dataset = args['dataset']

  with np.load("data/Hokuyo%d.npz"%dataset) as data:
    lidar_data = dict(data)

  with np.load(f"data/Odometry{dataset}_imu.npz") as data:
    odometry_data = dict(data)
    imu_odometry_poses = data["poses"]
    imu_odometry_stamps = data["stamps"]

  scan_matching = LaserScanMatching(lidar_data, odometry_data)
  plot_odometry([
    (scan_matching.imu_odometry_poses, scan_matching.imu_odometry_stamps, "IMU Odometry (motion model)"),
    (scan_matching.icp_odometry_poses, scan_matching.timestamps, "ICP Odometry (observation model)"),
  ])

  np.savez(f'data/Odometry{dataset}_icp.npz', poses=scan_matching.icp_odometry_poses, stamps=scan_matching.timestamps)
