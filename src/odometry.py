from pr2_utils import *

if __name__ == '__main__':
  
  args = parse_args(sys.argv)
  dataset = args['dataset']

  with np.load("data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps
    
    TICKS_PER_REVOLUTION = 360
    encoder_revoltions = encoder_counts / TICKS_PER_REVOLUTION
    FL_REV = np.cumsum(encoder_revoltions[1,:])
    FR_REV = np.cumsum(encoder_revoltions[0,:])
    RL_REV = np.cumsum(encoder_revoltions[3,:])
    RR_REV = np.cumsum(encoder_revoltions[2,:])

  with np.load("data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
    
    imu_omega = imu_angular_velocity[2,:].T

  odometry = DifferentialDriveOdometryPostProcess(
    (FL_REV + FR_REV)/2,
    (RL_REV + RR_REV)/2,
    encoder_stamps,
    imu_omega,
    imu_stamps
  )
  plot_odometry(
    [
      (odometry.poses, odometry.timestamps, "IMU Odometry (motion model)"),
    ],
    {"loc": "upper right"}
  )

  np.savez(f'data/Odometry{dataset}_imu.npz', poses=odometry.poses[1:], stamps=odometry.timestamps[1:])