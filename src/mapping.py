from pr2_utils import *
from scipy.interpolate import interp1d
import cv2
import open3d as o3d

if __name__ == '__main__':
  
  args = parse_args(sys.argv)
  dataset = args['dataset']
  odom_str = args['odom_str']

  with np.load("data/Hokuyo%d.npz"%dataset) as data:
    lidar_ranges = data["ranges"].T       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    angle_min = data["angle_min"] # start angle of the scan [rad]
    angle_max = data["angle_max"] # end angle of the scan [rad]
    angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    range_min = data["range_min"] # minimum range value [m]
    range_max = data["range_max"] # maximum range value [m]

  with np.load(f"data/Odometry{dataset}_{odom_str}.npz") as data:
    odometry = data["poses"]
    odometry_stamps = data["stamps"]

  map_resolution = 20
  map_size = np.array([1000, 1000])
  map_center= np.array([330, 330])

  occupancy_map = np.zeros(map_size, dtype=int)
  odoms = np.array([np.interp(lidar_stamps, odometry_stamps, odometry[:, i]) for i in range(odometry.shape[1])]).T
  bar= tqdm(range(lidar_stamps.shape[0]))
  for i in bar:
    pc_coord_body = lidarscan_to_pointcloud(lidar_ranges[i], range_min, range_max, angle_min, angle_max, angle_increment)
    
    odom = odoms[i]
    wTo = pose2d_to_transformation(odom)
    pc_coord_world = to_xyz_points(wTo @ (to_homogeneous_points(pc_coord_body)))
    pc_coord_map = map_resolution*pc_coord_world[:,:2] + map_center

    lines = []
    sx, sy = (map_resolution*odom[:2]+ map_center).astype(int)
    for point in pc_coord_map.astype(int):
        ex, ey = point.astype(int)
        lines.append(bresenham2D(sx, sy, ex, ey).T.astype(int))
    empty_points = np.unique(np.vstack(lines), axis=0, return_index=False)
    valid_mask = (0 <= empty_points[:, 0]) & (empty_points[:, 0] < occupancy_map.shape[1]) & \
        (0 <= empty_points[:, 1]) & (empty_points[:, 1] < occupancy_map.shape[0])
    filtered_points = empty_points[valid_mask]
    occupancy_map[filtered_points[:, 1], filtered_points[:, 0]] -= 1

    endpoints = pc_coord_map.astype(int)
    endpoint_mask = (0 <= endpoints[:, 0]) & (endpoints[:, 0] < occupancy_map.shape[1]) & \
                    (0 <= endpoints[:, 1]) & (endpoints[:, 1] < occupancy_map.shape[0])
    filtered_endpoints = endpoints[endpoint_mask]
    occupancy_map[filtered_endpoints[:, 1], filtered_endpoints[:, 0]] += 4

  np.savez(f'data/omap{dataset}_{odom_str}.npz', map=occupancy_map)

  probability_map = 1 / (1 + np.exp(-occupancy_map, dtype=np.float64))
  # Plotting
  plt.figure(figsize=(10, 8))
  plt.imshow(probability_map, cmap='viridis', origin='lower')
  plt.colorbar()
  # plt.title('Occupancy Probability Map')

  odometry_map = map_resolution*odometry[:,:2] + map_center
  plt.plot(odometry_map[:,0], odometry_map[:,1], color='white', linewidth=1, label=f'{odom_str} trajectory')
  plt.legend()
  plt.show(block=True)