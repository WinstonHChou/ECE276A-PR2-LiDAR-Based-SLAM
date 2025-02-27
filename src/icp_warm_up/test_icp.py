
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result, icp, o3d_icp, auto_guess_icp

if __name__ == "__main__":
  obj_names = ['drill', 'liq_container'] # drill or liq_container

  # obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds
  for obj_name in obj_names:
    source_pc = read_canonical_model(obj_name)

    for i in range(num_pc):
      target_pc = load_pc(obj_name, i)

      # estimated_pose, you need to estimate the pose with ICP
      pose = np.eye(4)

      if obj_name == "drill" and i == 0:
        pose, error = icp(source_pc, target_pc, pose) # normal icp
      else:
        pose, error = auto_guess_icp(source_pc, target_pc, trial=4) # auto guess icp

      # visualize the estimated result
      visualize_icp_result(source_pc, target_pc, pose)

