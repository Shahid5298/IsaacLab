from typing import Dict, List
import os 
import torch
import numpy as np
import math 

from bodex.util.sample_lib import HaltonGenerator
from bodex.util.tensor_util import normalize_vector
from bodex.util.logger import log_warn
from bodex.types.base import TensorDeviceType
from bodex.types.robot import RobotConfig
from bodex.types.math import Pose
from bodex.cuda_robot_model.cuda_robot_model import CudaRobotModel
from bodex.geom.basic_transform import euler_angles_to_matrix, matrix_from_rot_repre, matrix_to_quaternion
from bodex.geom.sdf.world import WorldCollision
from bodex.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from plyfile import PlyData, PlyElement
def save_pc_as_ply(pc, path):
    have_rgb = True if pc.shape[1] == 6 else False
    xyz = pc[:, :3]
    if have_rgb:
        rgb = (pc[:, 3:] * 255).astype(np.uint8)  # RGB 转换为整数
        vertex_data = np.empty(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    else:
        vertex_data = np.empty(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    if have_rgb:
        vertex_data['r'] = rgb[:, 0]
        vertex_data['g'] = rgb[:, 1]
        vertex_data['b'] = rgb[:, 2]
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def max_z_orthogonal(tensor):
    # 计算向量的范数平方
    norm_sq = torch.sum(tensor**2, dim=1, keepdim=True)
    # Z轴单位向量 (0,0,1)
    ez = torch.zeros_like(tensor)
    ez[:, 2] = 1.0
    # 计算向量的z分量
    v_z = tensor[:, 2:3]
    # 计算投影：ez - (v_z/norm_sq)*v
    proj = ez - (v_z / norm_sq) * tensor
    # 计算投影的范数
    proj_norm = torch.norm(proj, dim=1, keepdim=True)
    # 处理特殊情况：v与z轴平行的情况
    # 创建mask来标识投影范数接近0的向量
    mask = (proj_norm < 1e-8).squeeze()
    # 对于这些向量，使用(1,0,0)代替
    special_case = torch.zeros_like(tensor)
    special_case[:, 0] = 1.0
    # 对其他向量进行归一化
    normalized_proj = proj / proj_norm
    # 使用mask合并结果
    result = torch.where(mask.unsqueeze(1), special_case, normalized_proj)
    
    return result

class HeurGraspSeedGenerator:
    
    def __init__(self,
        seeder_cfg: Dict, 
        full_robot_model: CudaRobotModel,
        ik_solver: IKSolver,
        world_coll_checker: WorldCollision,
        obj_lst: List, 
        tensor_args: TensorDeviceType
    ):
        self.tensor_args = tensor_args
        all_dof = full_robot_model.dof
        use_root_pose = full_robot_model.use_root_pose
        self.full_robot_model = full_robot_model
        self.ik_solver = ik_solver
        self.tr_link_names = full_robot_model.transfered_link_name
        if ik_solver is not None:
            self.replace_ind = ik_solver.kinematics.kinematics_config.get_replace_index(full_robot_model.kinematics_config)
        
        if use_root_pose:
            assert len(self.tr_link_names) == 1
            self.q_dof = all_dof - 7
        else:
            if seeder_cfg['ik_init_q'] is None:
                self.ik_init = None 
            else:
                self.ik_init = tensor_args.to_device(seeder_cfg['ik_init_q']).view(1,1,-1)
            self.q_dof = all_dof
        
        # jitter r and t 
        if self.tr_link_names is not None:
            self.skip_transfer = seeder_cfg['skip_transfer']
            self.jitter_rt_random_gen = self._set_jitter_tr(seeder_cfg['jitter_dist'], seeder_cfg['jitter_angle'])
            
        self.seeder_cfg = seeder_cfg
        self.world_coll_checker = world_coll_checker
        self.reset(obj_lst)
        return 
    
    def _init_r_from_axis(self, r0):
        base_axis_palm = normalize_vector(r0)
        base_t1 = self.tensor_args.to_device([0, 1, 0])
        base_t2 = self.tensor_args.to_device([0, 0, 1]) # avoid base_axis_palm parallel to base_t1
        proj_xy = (base_t1 * base_axis_palm).sum(dim=-1, keepdim=True).abs()
        base_axis_thumb = torch.where(proj_xy > 0.99, base_t2, base_t1)
        r6d = torch.cat([base_axis_palm, base_axis_thumb], dim=-1)
        return r6d 
    
    def _set_base_trq(self, t, r, q, extra_info: WorldCollision=None):
        # log_warn(f'Initialize hand pose. t: {t}, r: {r}, q: {q}')
        if self.tr_link_names is None:
            base_t = None 
            base_r = None 
        else:
            tr_num = len(self.tr_link_names)
            if t is not None and r is not None:
                base_t = self.tensor_args.to_device(t).view(1, 1, tr_num, -1)
                r_repre = self.tensor_args.to_device(r).view(1, 1, tr_num, -1)
                ind_random_gen = None 
            elif t is None and r is None:
                # # original
                # base_t = extra_info.surface_sample_positions.view(-1, 1, tr_num, 3)
                # r_repre = self._init_r_from_axis(-extra_info.surface_sample_normals).view(-1, 1, tr_num, 6)

                # dual hands
                # base_t = extra_info.surface_sample_positions.view(-1, 1, tr_num, 3)  # [n, 1, 1, 3]

                # # symmetric
                # # 计算center_pos作为对称中心
                # center_pos = extra_info._contact_mesh_surface_points[0].mean(dim=-2)  # [3,]
                # # 计算base_t关于center_pos的对称点
                # symmetric_points = (2 * center_pos - base_t).squeeze().unsqueeze(1)  # [n, 1, 1, 3] -> [n, 1, 3]
                # all_points = base_t.clone().squeeze().unsqueeze(0)  # [n, 1, 1, 3] -> [1, n, 3]
                # # 计算所有点到对称点的距离
                # distances = torch.norm(all_points - symmetric_points, dim=-1, p=2)  # 
                # # 找到每个对称点的最近点的索引
                # indices = torch.argmin(distances, dim=1)  # 

                # # equal in height, symmetric in 2D
                # # 计算center_pos作为对称中心
                # center_pos = extra_info._contact_mesh_surface_points[0].mean(dim=-2)  # [3,]
                # # 计算base_t关于center_pos的等高2D对称点
                # symmetric_points = (2 * center_pos - base_t)
                # symmetric_points[..., 2] = base_t[..., 2]
                # symmetric_points = symmetric_points.squeeze().unsqueeze(1)  # [n, 1, 3]
                # all_points = base_t.clone().squeeze().unsqueeze(0)  # [n, 1, 1, 3] -> [1, n, 3]
                # # 计算所有点到对称点的距离
                # distances = torch.norm(all_points - symmetric_points, dim=-1, p=2)  # 
                # # 找到每个对称点的最近点的索引
                # indices = torch.argmin(distances, dim=1)  # 
                # # 获取最近的点作为base_t_1
                # base_t_1 = extra_info._contact_mesh_surface_points[0][indices].view(-1, 1, tr_num, 3)

                # equal-in-height points around the center, thumb up
                base_t_list = []
                base_t_1_list = []
                for obj_idx in range(extra_info._contact_mesh_surface_points.shape[0]):
                    center_pos = extra_info._contact_mesh_surface_points[obj_idx].mean(dim=-2)  # [3,]
                    num_point = extra_info._contact_mesh_surface_points[obj_idx].shape[0]
                    x_min = center_pos[1] - 0.07
                    x_max = center_pos[1] + 0.03
                    if extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() - extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() < 0.1:  # flat object
                        z_min = extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.15
                        z_max = extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() + 0.1
                    elif extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() - extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() < 0.35:  # small object
                        z_min = extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.15
                        z_max = torch.max(extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() - 0.1, extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.2)
                    else:
                        z_min = torch.max(extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.16, center_pos[2] - 0.04)
                        z_max = torch.min(extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() - 0.14, center_pos[2] + 0.06)
                    assert z_min < z_max, 'the object is too small, the height is smaller than 0.05'
                    y_low = extra_info._contact_mesh_surface_points[obj_idx][..., 1].min() - 0.06
                    y_high = extra_info._contact_mesh_surface_points[obj_idx][..., 1].max() + 0.06
                    x = torch.rand(num_point).to(self.tensor_args.device) * (x_max - x_min) + x_min
                    z = torch.rand(num_point).to(self.tensor_args.device) * (z_max - z_min) + z_min
                    base_t_list.append(torch.stack([x, y_low.repeat(num_point), z], dim=-1).unsqueeze(1).unsqueeze(1))  # right hand
                    base_t_1_list.append(torch.stack([x, y_high.repeat(num_point), z], dim=-1).unsqueeze(1).unsqueeze(1))  # left hand
                base_t = torch.cat(base_t_list)
                base_t_1 = torch.cat(base_t_1_list)

                # save_pc_as_ply(extra_info._contact_mesh_surface_points[0].cpu().numpy(), 'contact_mesh_surface_points.ply')

                # # original
                # r_repre = self._init_r_from_axis(-extra_info.surface_sample_normals).view(-1, 1, tr_num, 6)
                # r_repre_1 = self._init_r_from_axis(-extra_info.surface_sample_normals[indices]).view(-1, 1, tr_num, 6)
                # # thumb up
                # r_repre[..., 3:6] = max_z_orthogonal(r_repre[..., :3].reshape(-1, 3)).reshape(r_repre[..., 3:6].shape)
                # r_repre_1[..., 3:6] = max_z_orthogonal(r_repre_1[..., :3].reshape(-1, 3)).reshape(r_repre_1[..., 3:6].shape)
                # coordinate-aligned
                r_repre = torch.tensor([1.0, 0, 0, 0, 0, 1.0]).to(self.tensor_args.device).repeat(base_t.shape[0], 1, 1, 1)  # right hand
                r_repre_1 = torch.tensor([1.0, 0, 0, 0, 0, -1.0]).to(self.tensor_args.device).repeat(base_t_1.shape[0], 1, 1, 1)  # left hand

                ind_random_gen = HaltonGenerator(
                    len(extra_info.surface_sample_ind_upper), 
                    self.tensor_args, 
                    up_bounds=extra_info.surface_sample_ind_upper,
                    low_bounds=extra_info.surface_sample_ind_lower,
                    seed=1312)  
                assert tr_num == 1, "TODO: implement auto initialization for multiple hands"
            else:
                raise NotImplementedError
                
            base_r = matrix_from_rot_repre(r_repre)
            if 'r_repre_1' in locals():
                base_r_1 = matrix_from_rot_repre(r_repre_1)
    
        base_q = q if q is not None else [0]*self.q_dof
        assert len(base_q) == self.q_dof, self.q_dof
        base_q = self.tensor_args.to_device(base_q).view(1, 1, -1)
        if self.tr_link_names is not None and base_t.shape[0] > 1:
            base_q = base_q.expand(base_t.shape[0], base_t.shape[1], -1)
        if 'base_r_1' in locals():
            return base_t, base_r, base_t_1, base_r_1, base_q, ind_random_gen
        return base_t, base_r, base_q, ind_random_gen
    
    def _set_jitter_tr(self, jitter_dist, jitter_angle):
        jitter_bound_low = (jitter_dist[0] + [i/180*np.pi for i in jitter_angle[0]]) * len(self.tr_link_names)
        jitter_bound_up = (jitter_dist[1] + [i/180*np.pi for i in jitter_angle[1]]) * len(self.tr_link_names)
        random_gen = HaltonGenerator(
            len(jitter_bound_low), 
            self.tensor_args, 
            up_bounds=jitter_bound_up,
            low_bounds=jitter_bound_low,
            seed=1312)
        return random_gen
    
    def _load_base_trq(self, obj_lst, load_path_dict):
        robot_pose = []
        for obj_code in obj_lst:
            obj_code = obj_code.split('_scale_')[0] # This is only to fit the need of jialiang's data
            path = os.path.join(load_path_dict['base'], obj_code, load_path_dict['suffix'])
            log_warn(f'load hand pose initialization from {path}')
            data = dict(np.load(path, allow_pickle=True))
            tmp_robot_pose = self.tensor_args.to_device(data['robot_pose'])
            robot_pose.append(tmp_robot_pose)
        robot_pose = torch.stack(robot_pose, dim=0)
        if self.tr_link_names is None:
            base_t = None 
            base_r = None
        else:
            rot_repre_num = (robot_pose.shape[-1] - self.q_dof) // len(self.tr_link_names) - 3
            if len(self.tr_link_names) > 1:
                raise NotImplementedError
            base_t = robot_pose[..., :3].unsqueeze(-2)
            base_r = matrix_from_rot_repre(robot_pose[..., 3:3+rot_repre_num]).unsqueeze(-3)
        base_q = robot_pose[..., -self.q_dof:][..., load_path_dict['reorder_q']]
        return base_t, base_r, base_q
    
    def reset(self, obj_lst = None):
        self.jitter_rt_random_gen.reset()
        
        # init base r, t, q
        if self.seeder_cfg['load_path'] is not None:
            assert obj_lst is not None
            # [b, n, tr_num, 3], [b, n, tr_num, 3, 3], [b, n, q_dof] 
            self.base_t, self.base_r, self.base_q = self._load_base_trq(obj_lst, self.seeder_cfg['load_path'])
            self.ind_random_gen = None
        else:
            # [b/1, 1, tr_num, 3], [b/1, 1, tr_num, 3, 3], [b/1, 1, q_dof] 
            result_tuple = self._set_base_trq(self.seeder_cfg['t'], 
                                        self.seeder_cfg['r'], 
                                        self.seeder_cfg['q'], 
                                        extra_info=self.world_coll_checker
                                    )
            if len(result_tuple) == 6:
                self.base_t, self.base_r, self.base_t_1, self.base_r_1, self.base_q, self.ind_random_gen = result_tuple
            else:
                self.base_t, self.base_r, self.base_q, self.ind_random_gen = result_tuple
        return 
    
    def _jitter_on_base_tr(self, base_trans, base_rot):
        batch, num_samples, tr_num = base_trans.shape[:-1]
        rand_num = self.jitter_rt_random_gen.get_samples(batch*num_samples, bounded=True).view(batch, num_samples, tr_num, 6)
        rand_dist = rand_num[..., :3]
        rand_jitter_angle = rand_num[..., 3:]
        
        # calculate jittered translation and rotation
        jitter_rotation = euler_angles_to_matrix(torch.flip(rand_jitter_angle, [-1]), 'ZYX')
        final_rotation = base_rot @ jitter_rotation
        final_translation = base_trans - (final_rotation @ rand_dist.unsqueeze(-1)).squeeze(-1)
        return final_translation, final_rotation
    
    def _sample_to_shape(self, batch, num_samples):
        if self.ind_random_gen is not None:
            sample_idx = self.ind_random_gen.get_samples(num_samples, bounded=True).long()

            assert self.base_t.shape[0] % sample_idx.shape[1] == 0, 'self.base_t is not divisible by world_num'
            num_point_per_object = self.base_t.shape[0] // sample_idx.shape[1]
            sample_idx_for_obj0 = torch.randint(0, num_point_per_object, (sample_idx.shape[0],), device=sample_idx.device)
            sample_idx = torch.stack([sample_idx_for_obj0 + obj_idx * num_point_per_object for obj_idx in range(sample_idx.shape[1])], dim=1).long()

            base_rot = self.base_r[sample_idx].transpose(1, 0).squeeze(2) 
            base_trans = self.base_t[sample_idx].transpose(1, 0).squeeze(2) 
            base_q = self.base_q[sample_idx].transpose(1, 0).squeeze(2) 
            if hasattr(self, 'base_t_1') and hasattr(self, 'base_r_1'):
                base_rot_1 = self.base_r_1[sample_idx].transpose(1, 0).squeeze(2) 
                base_trans_1 = self.base_t_1[sample_idx].transpose(1, 0).squeeze(2) 
        else:
            if self.base_q.shape[0] < batch or self.base_q.shape[1] < num_samples: 
                repeat_b = math.ceil(batch / self.base_q.shape[0])
                repeat_n = math.ceil(num_samples / self.base_q.shape[1])
                self.base_r = self.base_r.repeat(repeat_b, repeat_n, 1, 1, 1) if self.base_r is not None else None 
                self.base_t = self.base_t.repeat(repeat_b, repeat_n, 1, 1) if self.base_t is not None else None
                self.base_q = self.base_q.repeat(repeat_b, repeat_n, 1)
            base_rot = self.base_r[:batch, :num_samples] if self.base_r is not None else None 
            base_trans = self.base_t[:batch, :num_samples] if self.base_t is not None else None
            base_q = self.base_q[:batch, :num_samples]
        if 'base_rot_1' in locals():
            return base_trans, base_rot, base_trans_1, base_rot_1, base_q
        return base_trans, base_rot, base_q
    
    def get_samples(self, batch, num_samples):
        result_tuple = self._sample_to_shape(batch, num_samples)
        if len(result_tuple) == 5:
            base_trans, base_rot, base_trans_1, base_rot_1, base_q = result_tuple
        else:
            base_trans, base_rot, base_q = result_tuple
        if self.tr_link_names is not None:
            # final_trans, final_rot = self._jitter_on_base_tr(base_trans, base_rot)
            final_trans, final_rot = base_trans, base_rot

            if not self.skip_transfer:
                final_trans, final_rot = self.full_robot_model.get_transfered_pose(final_trans.contiguous(), final_rot.contiguous(), self.tr_link_names)
            final_quat = matrix_to_quaternion(final_rot)
            if 'base_trans_1' in locals():
                # final_trans_1, final_rot_1 = self._jitter_on_base_tr(base_trans_1, base_rot_1)
                final_trans_1, final_rot_1 = base_trans_1, base_rot_1

                if not self.skip_transfer:
                    final_trans_1, final_rot_1 = self.full_robot_model.get_transfered_pose(final_trans_1.contiguous(), final_rot_1.contiguous(), self.tr_link_names)
                final_quat_1 = matrix_to_quaternion(final_rot_1)

        if self.ik_solver is not None:
            target_link_poses = {}
            for i, link_name in enumerate(self.tr_link_names):
                target_link_poses[link_name] = Pose(final_trans[..., i, :].reshape(-1, 3), final_quat[..., i, :].reshape(-1, 4))
                if i == 0:
                    goal = target_link_poses[link_name] 
            ik_init = self.ik_init.expand(batch, num_samples, -1) if self.ik_init is not None else None 
            result = self.ik_solver.solve_batch(goal, 
                                    seed_config=ik_init, 
                                    link_poses=target_link_poses
                                )
            if torch.any(~result.success):
                log_warn(f'ik result: {result.success.flatten()}')
            arm_q = result.solution.view(batch, num_samples, -1)
            hand_pose = base_q
            hand_pose[..., self.replace_ind] = arm_q
        else:
            # use_root_pose is True
            # hand_pose = torch.cat([final_trans.squeeze(-2), final_quat.squeeze(-2), base_q], dim=-1) 

            # single hand
            # _, b, _, d = final_quat.shape
            # final_rot = np.zeros([1, b, 3])

            # from scipy.spatial.transform import Rotation as R
            # for i in range(b):
            #     quat_numpy = final_quat[0, i, 0, :4].cpu().numpy()
            #     final_rot[0, i, :3] = R.from_quat(quat_numpy, scalar_first=True).as_euler('XYZ', degrees=False)

            # final_rot = self.tensor_args.to_device(final_rot)
            # hand_pose = torch.cat([final_trans.squeeze(-2), final_rot, base_q[:, :, 6:]], dim=-1)

            # dual hand
            # _, b, _, d = final_quat.shape
            # final_rot = np.zeros([1, b, 3])
            # final_rot_1 = np.zeros([1, b, 3])

            # from scipy.spatial.transform import Rotation as R
            # for i in range(b):
            #     quat_numpy = final_quat[0, i, 0, :4].cpu().numpy()
            #     final_rot[0, i, :3] = R.from_quat(quat_numpy, scalar_first=True).as_euler('XYZ', degrees=False)
            #     quat_numpy_1 = final_quat_1[0, i, 0, :4].cpu().numpy()
            #     final_rot_1[0, i, :3] = R.from_quat(quat_numpy_1, scalar_first=True).as_euler('XYZ', degrees=False)
            #     # rot_in_urdf = R.from_quat(quat, scalar_first=True).as_euler('XYZ', degrees=False)
            #     # quat = R.from_euler('XYZ', rot_in_urdf).as_quat(scalar_first=True)
            # final_rot = self.tensor_args.to_device(final_rot)
            # final_rot_1 = self.tensor_args.to_device(final_rot_1)
            # hand_pose = torch.cat([final_trans.squeeze(-2), final_rot, base_q[:, :, 6:10], final_trans_1.squeeze(-2), final_rot_1, base_q[:, :, 16:]], dim=-1)

            random_rot_Z = -torch.pi*7/16 * torch.rand([base_q.shape[0], base_q.shape[1]]).to(self.tensor_args.device)
            final_rot = torch.tensor([[[torch.pi*15/32, torch.pi*1/32, 0]]]).to(self.tensor_args.device).repeat(base_q.shape[0], base_q.shape[1], 1)
            final_rot[:, :, 2] = random_rot_Z
            final_rot_1 = torch.tensor([[[-torch.pi*15/32, torch.pi*1/32, 0]]]).to(self.tensor_args.device).repeat(base_q.shape[0], base_q.shape[1], 1)
            final_rot_1[:, :, 2] = -random_rot_Z

            if base_q.shape[-1] == 44:  # leap
                hand_pose = torch.cat([final_trans_1.squeeze(-2), final_rot_1, base_q[:, :, 6:10], final_trans.squeeze(-2), final_rot, base_q[:, :, 16:]], dim=-1)
            elif base_q.shape[-1] == 36:  # xhand
                hand_pose = torch.cat([final_trans.squeeze(-2), final_rot, base_q[:, :, 6:18], final_trans_1.squeeze(-2), final_rot_1, base_q[:, :, 24:36]], dim=-1)
            else :
                print(f'[ERROR] hand pose shape {base_q.shape[-1]} is not supported')

        return hand_pose
    