import torch
import torch.nn.functional as F
from torch import nn
# from helpers import rot2qua_torch, qua2rot_torch
# from helpers import pcd_extrinsic_transform_torch
    
class realignment_layer(nn.Module):
    def __init__(self, crop_pcd = False):
        super(realignment_layer, self).__init__()
        self.rotate_pcd = pcd_extrinsic_transform_torch(crop = crop_pcd)

    def forward(self, pcd_mis, T_mis, delta_q_pred, delta_t_pred):
        device_ = delta_q_pred.device
        batch_size = delta_q_pred.shape[0]
        batch_T_pred = torch.tensor([]).to(device_)
        batch_pcd_realigned = []

        # print("pcd: ",pcd_mis.shape, "T_mis: ", T_mis.shape, "q: ", delta_q_pred.shape, "t: ", delta_t_pred.shape)

        for i in range(batch_size):
            delta_R_pred = qua2rot_torch(delta_q_pred[i])
            delta_tr_pred = torch.reshape(delta_t_pred[i],(3,1))
            delta_T_pred = torch.hstack((delta_R_pred, delta_tr_pred)) 
            delta_T_pred = torch.vstack((delta_T_pred, torch.Tensor([0., 0., 0., 1.]).to(device_)))

            T_act_pred = torch.unsqueeze(torch.matmul(torch.linalg.inv(delta_T_pred), T_mis[i]), 0)

            # print(torch.linalg.inv(delta_T_pred).shape)
            # print(pcd_mis[i].shape)
            pcd_pred = self.rotate_pcd(pcd_mis[i], torch.linalg.inv(delta_T_pred))

            batch_T_pred = torch.cat((batch_T_pred, T_act_pred), 0)
            batch_pcd_realigned.append(pcd_pred)
        
        return batch_T_pred, batch_pcd_realigned

class pcd_extrinsic_transform_torch: # transform PCD into fisheye camera reference frame.
    def __init__(self, crop = True):
        self.crop = crop

    def __call__(self, point_cloud, T_ext):
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        device_ = point_cloud.device
        n_points = point_cloud.shape[0]
        pcd_fisheye = torch.matmul(T_ext, torch.hstack((point_cloud, torch.ones(n_points, 1).to(device_))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        if self.crop:
            condition = (z_axis>=0)
            new_pcd = pcd_fisheye[condition]
        else:
            new_pcd = pcd_fisheye

        # print(point_cloud.shape)
        # print(new_pcd.shape)

        return new_pcd

def qua2rot_torch(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [3x3] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((3, 3), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    
    return mat