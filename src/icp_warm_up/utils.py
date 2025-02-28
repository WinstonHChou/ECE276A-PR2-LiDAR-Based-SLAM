import os
import scipy.io as sio
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def read_canonical_model(model_name):
  '''
  Read canonical model from .mat file
  model_name: str, 'drill' or 'liq_container'
  return: numpy array, (N, 3)
  '''
  model_fname = os.path.join('./data', model_name, 'model.mat')
  model = sio.loadmat(model_fname)

  cano_pc = model['Mdata'].T / 1000.0 # convert to meter

  return cano_pc


def load_pc(model_name, id):
  '''
  Load point cloud from .npy file
  model_name: str, 'drill' or 'liq_container'
  id: int, point cloud id
  return: numpy array, (N, 3)
  '''
  pc_fname = os.path.join('./data', model_name, '%d.npy' % id)
  pc = np.load(pc_fname)

  return pc


def visualize_icp_result(source_pc, target_pc, pose):
  '''
  Visualize the result of ICP
  source_pc: numpy array, (N, 3)
  target_pc: numpy array, (N, 3)
  pose: SE(4) numpy array, (4, 4)
  '''
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  o3d.visualization.draw_geometries([source_pcd, target_pcd])

def to_homogeneous_points(xyz_points: np.ndarray) -> np.ndarray:
  if xyz_points.shape[1] != 3:
        raise ValueError("xyz_points must have shape (n, 3)")
  return np.column_stack((xyz_points, np.ones(xyz_points.shape[0]))).T

def to_xyz_points(homogeneous_points: np.ndarray) -> np.ndarray:
  if homogeneous_points.shape[0] != 4:
        raise ValueError("homogenous_points must have shape (n, 4)")
  return homogeneous_points[:3, :].T

def to_transformation(R: np.ndarray, p: np.ndarray):
  error_msg = ""
  if R.shape != (3,3):
    error_msg += "R must be (3, 3).\n"
  if p.shape != (3,1):
    error_msg += "p must be (3, 1).\n"
  if error_msg:
    raise ValueError(error_msg)

  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3:] = p

  return T

def to_R_p(T: np.ndarray):
  if T.shape != (4,4):
    raise ValueError("T must be (4, 4).\n")

  R = T[:3, :3]
  p = T[:3, 3:]

  return R, p

def find_closest_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
  """
  Find the closest points in the target points for each point in the source using Scikit-learn's NearestNeighbors.
  """
  neigh = NearestNeighbors(n_neighbors=1)
  neigh.fit(target)
  indices = neigh.kneighbors(source, return_distance=False)
  closest_points = target[indices.squeeze()]
  return closest_points

def find_closest_points_threshold(source, target, percentile):
  """
  Find the closest points in the target for each point in the source using Scikit-learn's NearestNeighbors.
  """
  neigh = NearestNeighbors(n_neighbors=1)
  neigh.fit(target)
  distances, indices = neigh.kneighbors(source, return_distance=True)

  distance_threshold = np.percentile(distances, percentile * 100)
  mask = distances.flatten() <= distance_threshold

  closest_points = target[indices.flatten()]
  return closest_points, mask

def estimate_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
  """
  Given (source, target) point association,
  Estimate the rotation and translation using Kabsch Algorithm.
  """
  source_centroid = np.mean(source, axis=0)
  target_centroid = np.mean(target, axis=0)
  Q =  np.dot((target - target_centroid).T, (source - source_centroid))
  U, _, Vt = np.linalg.svd(Q)
  R = U @ Vt
  # Convert LHS to RHS if found R to be LHS
  if np.linalg.det(R) < 0:
      Vt[2,:] *= -1
      R = U @ Vt
  p = (target_centroid - R @ source_centroid).reshape(3,1)
  return to_transformation(R, p)

def icp(source_pc: np.ndarray, target_pc: np.ndarray, T_0 = np.eye(4), iterations=100, tolerance=1e-8):
  '''
  return estimated_transformation and error
  '''
  source_pc_transformed = to_xyz_points(T_0 @ to_homogeneous_points(source_pc))

  prev_error = float('inf')
  for i in range(iterations):
    closest_points = find_closest_points(source_pc_transformed, target_pc)
    T = estimate_transformation(source_pc_transformed, closest_points)
    source_pc_transformed = to_xyz_points(T @ to_homogeneous_points(source_pc_transformed))
    error = np.sqrt(np.mean(np.sum((closest_points - source_pc_transformed) ** 2, axis=1))) # RSME
    if np.abs(prev_error - error) < tolerance:
        break
    prev_error = error
  T = estimate_transformation(source_pc, closest_points)
  return T, error

def o3d_icp(source_pc, target_pc, T_0 = np.eye(4), threshold = 5):
  source = o3d.geometry.PointCloud()
  target = o3d.geometry.PointCloud()
  source.points = o3d.utility.Vector3dVector(source_pc)
  target.points = o3d.utility.Vector3dVector(target_pc)
  # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  
  # Apply ICP registration
  reg_p2p = o3d.pipelines.registration.registration_icp(
      source, target, threshold, T_0,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
      # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
      o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1e-8, max_iteration=1000))

  return reg_p2p.transformation, reg_p2p.inlier_rmse

def auto_guess_icp(source, target, trial = 10, icp_type="icp"):
    best_T, best_error = np.eye(4), float("inf")
    for yaw in np.linspace(0, 2*np.pi, trial):
        R_0 = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0,0,1]
        ])
        p_0 = np.array([1,0,0]).reshape(3,1)
        
        if icp_type == "o3d_icp":
          T, error = o3d_icp(source, target, to_transformation(R_0, p_0))
        else:
          T, error = icp(source, target, to_transformation(R_0, p_0))
        if error < best_error:
            best_T = T
    return best_T, best_error

def icp_percentile(source, target, T_0 = np.eye(4), percentile=0.9, iterations=100, tolerance=1e-8):
  '''
  return estimated_transformation and error
  '''
  source_transformed = to_xyz_points(T_0 @ to_homogeneous_points(source))

  prev_error = float('inf')
  for i in range(iterations):
      closest_points, mask = find_closest_points_threshold(source_transformed, target, percentile)
      T = estimate_transformation(source_transformed[mask], closest_points[mask])
      source_transformed = to_xyz_points(T @ to_homogeneous_points(source_transformed))
      error = np.sqrt(np.mean(np.sum((closest_points - source_transformed) ** 2, axis=1))) # RSME
      if np.abs(prev_error - error) < tolerance:
          break
      prev_error = error
  T = estimate_transformation(source[mask], closest_points[mask])
  return T, error