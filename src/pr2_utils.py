
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
import sys
import scipy
from transforms3d.euler import mat2euler
from tqdm import tqdm

from icp_warm_up.utils import to_homogeneous_points, to_xyz_points, to_R_p, to_transformation, visualize_icp_result, icp, o3d_icp, icp_percentile

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

def parse_args(argv):
  parsed_args = {}
  for arg in argv:
    if '=' in arg:
      key, value = arg.split('=', 1)
      if key == 'dataset':
          parsed_args['dataset'] = int(value)
      # elif key == 'iterations':
      #     parsed_args['iterations'] = int(value)
      # elif key == 'stepSize':
      #     parsed_args['stepSize'] = float(value)
      # elif key == 'rough':
      #     parsed_args['rough'] = value.lower() in ['true', '1', 'yes']
  
  # Set default values if not provided
  parsed_args.setdefault('dataset', None)
  # parsed_args.setdefault('iterations', 300)   # Default Iterations: 300
  # parsed_args.setdefault('stepSize', 0.025)   # Default Step Size: 0.025
  # parsed_args.setdefault('rough', False)
  
  return parsed_args

class DifferentialDriveOdometryPostProcess:
  def __init__(self, left_encoder_rev, right_encoder_rev, encoder_stamps, imu_omega, imu_stamps, T_0 = np.eye(4)):

    WHEEL_DIAMETER = 0.254 # meters
    
    self.dt = np.diff(encoder_stamps)

    self.VL = np.diff(left_encoder_rev) * np.pi * WHEEL_DIAMETER / self.dt
    self.VR = np.diff(right_encoder_rev) * np.pi * WHEEL_DIAMETER / self.dt
    self.V = (self.VL + self.VR) / 2

    self.timestamps = np.cumsum(np.concatenate(([0], self.dt)))   # normalized timestamp
    imu_cum_stamps = np.cumsum(np.concatenate(([0], np.diff(imu_stamps)))) # normalized timestamp
    self.omega = np.interp(self.timestamps, imu_cum_stamps, imu_omega)

    self.poses = np.zeros((len(self.dt) + 1, 3))
    T_prev = T_0
    for i, pose in enumerate(self.poses):
      dt = self.dt[i-1]
      v_t = self.V[i-1]
      omega_t = self.omega[i-1]
  
      twist = np.zeros((4, 4))
      twist[0, 1] = -omega_t
      twist[1, 0] = omega_t
      twist[0, 3] = v_t
  
      T = T_prev @ scipy.linalg.expm(twist * dt)

      R = T[:3, :3]
      _, _, yaw = mat2euler(R, axes='sxyz')
  
      pose[0] = T[0, 3]             # x
      pose[1] = T[1, 3]             # y
      pose[2] = angle_modulus(yaw)  # theta
      
      T_prev = T

  def plot_velocity(self):
    plt.figure()
    plt.plot(self.VL, marker='o', linestyle='-', markersize=1, label="VL")
    plt.plot(self.VR, marker='o', linestyle='-', markersize=1, label="VR")
    plt.plot(self.V, marker='o', linestyle='-', markersize=1, label="V_average")
    plt.xlabel("number of data")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Comparison")
    plt.grid()
    plt.legend()
    plt.show(block=True)

class LaserScanMatching:
  def __init__(self, lidar_data, imu_odometry_data):
    self.lidar = lidar_data
    self.lidar_angle_min = lidar_data["angle_min"] # start angle of the scan [rad]
    self.lidar_angle_max = lidar_data["angle_max"] # end angle of the scan [rad]
    self.lidar_angle_increment = lidar_data["angle_increment"] # angular distance between measurements [rad]
    self.lidar_range_min = lidar_data["range_min"] # minimum range value [m]
    self.lidar_range_max = lidar_data["range_max"] # maximum range value [m]
    self.lidar_ranges = lidar_data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    self.lidar_stamps = lidar_data["time_stamps"]  # acquisition times of the lidar scans

    self.lidar_num_of_points = int((self.lidar_angle_max - self.lidar_angle_min) / self.lidar_angle_increment) + 1
    self.lidar_angles = np.linspace(self.lidar_angle_min, self.lidar_angle_max, self.lidar_num_of_points)
    
    self.imu_odometry_poses = imu_odometry_data["poses"]
    self.imu_odometry_stamps = imu_odometry_data["stamps"]

    self.timestamps = np.cumsum(np.concatenate(([0], np.diff(self.lidar_stamps))))         # normalized timestamp
    self.imu_odometry_stamps = np.cumsum(np.concatenate(([0], np.diff(self.imu_odometry_stamps)))) # normalized timestamp
    self.imu_odometry_poses_sync = np.array([np.interp(self.timestamps, self.imu_odometry_stamps, self.imu_odometry_poses[:,i]) for i in range(self.imu_odometry_poses.shape[1])]).T

    # print(self.imu_odometry_poses_sync[1400])
    # self.plot_laserscan(self.lidar_ranges.T[1400], self.imu_odometry_poses_sync[1400])

    self.icp_odometry_poses = self.icp_scan_matching()

  def lidarscan_to_pointcloud(self, ranges):
    '''
    return pointcloud in local frame
    '''
    range_indices = (ranges > self.lidar_range_min) & (ranges < self.lidar_range_max)
    z_offset = 0
    x = ranges[range_indices] * np.cos(self.lidar_angles[range_indices])
    y = ranges[range_indices] * np.sin(self.lidar_angles[range_indices])
    z = np.full(x.shape, z_offset)

    return np.stack([x,y,z]).T
  
  def icp_scan_matching(self):
    icp_T = np.zeros([self.timestamps.shape[0],4,4])
    icp_T[0] = np.eye(4)
    imu_odometry_T = pose2d_to_transformation(self.imu_odometry_poses_sync)
    error_sum = 0
    for i in tqdm(range(1, self.timestamps.shape[0])):
        T_guess = np.linalg.inv(imu_odometry_T[i-1]) @ imu_odometry_T[i]
        T, error = icp_percentile(
          self.lidarscan_to_pointcloud(self.lidar_ranges.T[i, :]),
          self.lidarscan_to_pointcloud(self.lidar_ranges.T[i-1, :]),
          T_guess
        )
        error_sum += error
        icp_T[i] = icp_T[i-1] @ T
    print(error_sum / self.timestamps.shape[0])
    icp_odometry = transformation_to_pose2d(icp_T)
    # print(icp_odometry)

    return icp_odometry
  
  def plot_laserscan(self, lidar_ranges, odometry_pose):
    odometry_transformation = pose2d_to_transformation(odometry_pose)
    pc = self.lidarscan_to_pointcloud(lidar_ranges)
    pc_to_odom = to_xyz_points(odometry_transformation @ (to_homogeneous_points(pc)))
    plt.scatter(pc_to_odom[:,0], pc_to_odom[:,1])

    dx = 3 * np.cos(odometry_pose[2])
    dy = 3 * np.sin(odometry_pose[2])
    plt.arrow(odometry_pose[0], odometry_pose[1], dx, dy, head_width=0.1, head_length=0.15, fc='red', ec='red')
    
    dx = 3 * np.cos(odometry_pose[2] + self.lidar_angle_max)
    dy = 3 * np.sin(odometry_pose[2] + self.lidar_angle_max)
    plt.arrow(odometry_pose[0], odometry_pose[1], dx, dy, head_width=0.1, head_length=0.15, fc='grey', ec='grey')
    
    dx = 3 * np.cos(odometry_pose[2] + self.lidar_angle_min)
    dy = 3 * np.sin(odometry_pose[2] + self.lidar_angle_min)
    plt.arrow(odometry_pose[0], odometry_pose[1], dx, dy, head_width=0.1, head_length=0.15, fc='grey', ec='grey')
    
    plt.grid(True)
    plt.show(block=True)

def angle_modulus(angle_rad):
    single_angle = not isinstance(angle_rad, np.ndarray)
    angle_rad = np.atleast_1d(angle_rad)
    
    angle_rad = np.mod(angle_rad + np.pi, 2*np.pi) - np.pi
    
    return angle_rad[0] if single_angle else angle_rad

def pose2d_to_transformation(pose2d):
  single_pose2d = False
  if pose2d.shape == (3,):
    single_pose2d = True
    pose2d = pose2d.reshape(1,3)

  transformation = np.zeros((pose2d.shape[0], 4, 4))
  transformation[:, 0, 0] = np.cos(pose2d[:, 2])
  transformation[:, 0, 1] = -np.sin(pose2d[:, 2])
  transformation[:, 1, 0] = np.sin(pose2d[:, 2])
  transformation[:, 1, 1] = np.cos(pose2d[:, 2])
  transformation[:, 2, 2] = 1
  transformation[:, 0, 3] = pose2d[:, 0]
  transformation[:, 1, 3] = pose2d[:, 1]
  transformation[:, 3, 3] = 1

  if single_pose2d:
    return transformation[0,:,:]

  return transformation

def transformation_to_pose2d(transformation):
  single_T = False
  if transformation.shape == (4,4):
    single_T = True
    transformation = transformation.reshape(1,4,4)

  R = transformation[:, :3, :3]
  euler_angles = np.array([mat2euler(R_i, axes='sxyz') for R_i in R])
  yaw = euler_angles[:,2]

  pose2d = np.zeros((transformation.shape[0], 3))
  pose2d[:, 0] = transformation[:, 0, 3]  # x
  pose2d[:, 1] = transformation[:, 1, 3]  # y
  pose2d[:, 2] = angle_modulus(yaw)       # theta

  if single_T:
    return pose2d[0,:]

  return pose2d

def plot_odometry(odometry_serious, legend_kw = {}):
    '''
    odometries: [(odometry1, timestamp1, label), (odometry2, timestamp2, label), ...]
    '''
    for serious in odometry_serious:
        odometry, stamp, label = serious[:3]
        kwargs = {}
        if len(serious) == 4:
            kwargs = serious[3]
        plt.plot(odometry[:,0], odometry[:,1], label=label, **kwargs)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(**legend_kw)
    plt.grid(True)
    plt.show(block=True)

    for i, serious in enumerate(odometry_serious):
        odometry, stamp, label = serious[:3]
        kwargs = {}
        if len(serious) == 4:
            kwargs = serious[3]
        plt.plot(stamp, odometry[:,0], label=label, **kwargs)
    plt.xlabel("Timestamp (sec)")
    plt.ylabel("X (m)")
    plt.legend(**legend_kw)
    plt.show(block=True)

    for i, serious in enumerate(odometry_serious):
        odometry, stamp, label = serious[:3]
        kwargs = {}
        if len(serious) == 4:
            kwargs = serious[3]
        plt.plot(stamp, odometry[:,1], label=label, **kwargs)
    plt.xlabel("Timestamp (sec)")
    plt.ylabel("Y (m)")
    plt.legend(**legend_kw)
    plt.show(block=True)

    for i, serious in enumerate(odometry_serious):
        odometry, stamp, label = serious[:3]
        kwargs = {}
        if len(serious) == 4:
            kwargs = serious[3]
        plt.plot(stamp, np.unwrap(odometry[:,2]), label=label, **kwargs)
    plt.xlabel("Timestamp")
    plt.ylabel("Heading (Yaw, radians)")
    plt.legend(**legend_kw)
    plt.show(block=True)

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))


def show_lidar():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("test_ranges.npy")
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()


def plot_map(mapdata, cmap="binary"):
  plt.imshow(mapdata.T, origin="lower", cmap=cmap)

def test_map():
  # Initialize a grid map
  MAP = {}
  MAP['res'] = np.array([0.05, 0.05])    # meters
  MAP['min'] = np.array([-20.0, -20.0])  # meters
  MAP['max'] = np.array([20.0, 20.0])    # meters
  MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
  isEven = MAP['size']%2==0
  MAP['size'][isEven] = MAP['size'][isEven]+1 # Make sure that the map has an odd size so that the origin is in the center cell
  MAP['map'] = np.zeros(MAP['size'])
  
  # Load Lidar scan
  ranges = np.load("test_ranges.npy")
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  valid1 = np.logical_and((ranges < 30),(ranges> 0.1))
  
  # Lidar points in the sensor frame
  points = np.column_stack((ranges*np.cos(angles), ranges*np.sin(angles)))
  
  # Convert from meters to cells
  cells = np.floor((points - MAP['min']) / MAP['res']).astype(int)
  
  # Insert valid points in the map
  valid2 = np.all((cells >= 0) & (cells < MAP['size']),axis=1)
  MAP['map'][tuple(cells[valid1&valid2].T)] = 1

  # Plot the Lidar points
  fig1 = plt.figure()
  plt.plot(points[:,0],points[:,1],'.k')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Lidar scan')
  plt.axis('equal')
  
  # Plot the grid map
  fig2 = plt.figure()
  plot_map(MAP['map'],cmap='binary')
  plt.title('Grid map')
  
  plt.show()

if __name__ == '__main__':
  show_lidar()
  test_bresenham2D()
  test_map()
  
  
  

