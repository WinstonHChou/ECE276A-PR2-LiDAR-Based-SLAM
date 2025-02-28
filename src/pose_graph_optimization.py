from pr2_utils import *

if __name__ == '__main__':
  
  args = parse_args(sys.argv)
  dataset = args['dataset']

  with np.load("data/Hokuyo%d.npz"%dataset) as data:
    lidar_data = dict(data)

  with np.load(f"data/Odometry{dataset}_imu.npz") as data:
    imu_odometry_data = dict(data)

  with np.load(f"data/Odometry{dataset}_icp.npz") as data:
    icp_odometry_data = dict(data)

  pose_graph_optimization = PoseGraphOptimization(lidar_data, imu_odometry_data, icp_odometry_data)
  plot_odometry([
    (pose_graph_optimization.imu_odometry_poses, pose_graph_optimization.imu_odometry_stamps, "IMU Odometry (motion model)"),
    (pose_graph_optimization.icp_odometry_poses, pose_graph_optimization.icp_odometry_stamps, "ICP Odometry (observation model)"),
    (pose_graph_optimization.factor_graph_optimized, pose_graph_optimization.timestamps, "factor_graph_optimized", {"linestyle":'--'})
  ])

  # np.savez(f'data/Odometry{dataset}_icp.npz', poses=pose_graph_optimization.icp_odometry_poses, stamps=pose_graph_optimization.timestamps)
