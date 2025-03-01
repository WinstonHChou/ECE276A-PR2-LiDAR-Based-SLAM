import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from pr2_utils import plot_odometry

datasets = [20, 21]
odom_strs = ["imu", "icp", "lc0"]#, "lc1"]

if __name__ == "__main__":
  for dataset in datasets:
    odom_str = "imu"
    with np.load(f"data/Odometry{dataset}_{odom_str}.npz") as data:
      imu_odometry = data["poses"]
      imu_odometry_stamp = data["stamps"]

    odom_str = "icp"
    with np.load(f"data/Odometry{dataset}_{odom_str}.npz") as data:
      icp_odometry = data["poses"]
      icp_odometry_stamp = data["stamps"]

    odom_str = "lc2"
    with np.load(f"data/Odometry{dataset}_{odom_str}.npz") as data:
      lc2_odometry = data["poses"]
      lc2_odometry_stamp = data["stamps"]

    # odom_str = "lc1"
    # with np.load(f"data/Odometry{dataset}_{odom_str}.npz") as data:
    #   lc1_odometry = data["poses"]
    #   lc1_odometry_stamp = data["stamps"]

    plot_odometry([
      (imu_odometry, imu_odometry_stamp, "IMU Odometry (motion model)", {"linewidth":1}),
      (icp_odometry, icp_odometry_stamp, "ICP Odometry (observation model)",  {"linewidth":1}),
      (lc2_odometry, lc2_odometry_stamp, "factor graph optimized", {"linewidth":1, "linestyle":'--'}),
      # (lc1_odometry, lc1_odometry_stamp, "loop closure optimized", {"linewidth":1, "linestyle":'--'}),
    ], {"loc":'lower center'})


