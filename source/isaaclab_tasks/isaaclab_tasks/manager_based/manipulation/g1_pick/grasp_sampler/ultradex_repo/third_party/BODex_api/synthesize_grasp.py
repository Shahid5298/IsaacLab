import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from bodex.geom.sdf.world import WorldConfig
from bodex.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from bodex.util.world_cfg_generator import get_world_config_dataloader
from bodex.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
    load_json
)

def pos_quat_to_mat(pos_quat):
    pos = pos_quat[:3]
    quat = pos_quat[3:]
    mat = np.eye(4) 
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return mat

class GraspSynthesizer:
    def __init__(self, config_path):
        self.manip_config_data = load_yaml(config_path)
        self.grasp_solver = None

    def synthesize_grasp(self, object_path, object_pose, object_scale):
        if type(object_pose) is not list:
            object_pose = object_pose.tolist()
        obj_code = object_path.split('/')[-1]
        full_path = f'{object_path}/mesh/simplified.obj'
        manip_name = f"{obj_code}_scale{str(int(object_scale * 100)).zfill(3)}"

        world_cfg = {'cuboid': {'table': {'dims': [2.0, 2.0, 0.2], 'pose': [0.0, 0.0, 0.0, 1, 0, 0, 0.0]}}, 'mesh': {}}
        world_cfg["mesh"][manip_name] = {
            "scale": object_scale,
            "pose": object_pose,
            "file_path": full_path,
            "urdf_path": f'{object_path}/urdf/coacd.urdf',
        }

        json_data = load_json(f'{object_path}/info/simplified.json')

        world_info_dict = {
            "world_cfg": [world_cfg],
            "obj_code": [obj_code],
            "manip_name": [manip_name],
            "obj_path": [full_path],
            "obj_scale": torch.tensor([object_scale], dtype=torch.float64),
            "obj_pose": torch.tensor([object_pose], dtype=torch.float64),
            "obj_gravity_center": torch.from_numpy(np.array([object_pose[:3] + R.from_quat(object_pose[3:], scalar_first=True).as_matrix() @ json_data["gravity_center"] * object_scale])).to(torch.float64),
            "obj_obb_length": torch.tensor([object_scale * np.linalg.norm(json_data["obb"]) / 2], dtype=torch.float64),
        }

        # get object pose, scale, and mesh. then calculate the z_min to set the height of the table
        object_mesh_path = world_info_dict['obj_path'][0]
        object_pose = world_info_dict['obj_pose'][0].cpu().numpy()
        object_pose = pos_quat_to_mat(object_pose)
        object_scale = world_info_dict['obj_scale'][0].item()
        object_mesh = trimesh.load(object_mesh_path)
        object_mesh.apply_transform(object_pose)
        object_mesh.apply_scale(object_scale)
        z_min = object_mesh.bounds[0][2]
        world_info_dict['world_cfg'][0]['cuboid']['table']['pose'][2] = z_min - world_info_dict['world_cfg'][0]['cuboid']['table']['dims'][2] / 2 + 0.01  # raise the table by 1cm for better collision avoidance

        if self.grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                        world_model=world_info_dict['world_cfg'],
                        manip_name_list=world_info_dict['manip_name'],
                        manip_config_data=self.manip_config_data,
                        obj_gravity_center=world_info_dict['obj_gravity_center'],
                        obj_obb_length=world_info_dict['obj_obb_length'],
                        use_cuda_graph=False,
                        store_debug=False,
                    )
            self.grasp_solver = GraspSolver(grasp_config)
        else:
            world_model = [WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict['world_cfg']]
            self.grasp_solver.update_world(world_model, world_info_dict['obj_gravity_center'], world_info_dict['obj_obb_length'], world_info_dict['manip_name'])

        result = self.grasp_solver.solve_batch_env(return_seeds=self.grasp_solver.num_seeds)

        squeeze_pose_qpos = torch.cat([result.solution[..., 1, :7], result.solution[..., 1, 7:] * 2 - result.solution[..., 0, 7:]], dim=-1)
        all_hand_pose_qpos = torch.cat([result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2)

        return all_hand_pose_qpos


if __name__ == "__main__":
    grasp_synthesizer = GraspSynthesizer('/cpfs/user/yangsizhe/project/common/BODex_api/src/bodex/content/configs/manip/sim_leap/fc.yml')
    grasp_pose = grasp_synthesizer.synthesize_grasp('/cpfs/user/yangsizhe/project/common/BODex_api/src/bodex/content/assets/object/DGN_obj/bidexgrasp_data/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924', np.array([0., 0., 0., 1., 0., 0., 0.]), 0.08)
