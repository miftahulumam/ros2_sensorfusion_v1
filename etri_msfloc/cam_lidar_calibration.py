#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from realsense2_camera_msgs.msg import Extrinsics
from .calib_inference import point_cloud2 as pc2
from .calib_inference.build_modified_vit_transcalib_lvt_efficientnet_demo import TransCalib_lvt_efficientnet_june2 as Model

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms as T
import cv2
from cv_bridge import CvBridge
import math
from dotwiz import DotWiz

class camera_lidar_calibration(Node):
    def __init__(self):
        super().__init__("camera_lidar_calibration")

        self.projection_publisher_ = self.create_publisher(
            Image,
            "/pcd_projection",
            10
        )

        self.rgb_subscriber_ = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.rgb_callback,
            10
        )

        self.pcd_subscriber_ = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.pcd_callback,
            10
        )

        self.intrinsics_subscriber_ = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.intrinsics_callback,
            10
        )
        
        # self.extrinsic_subscriber_ = self.create_subscription(
        #     Extrinsics,
        #     "/camera/extrinsics/depth_to_color",
        #     self.extrinsic_callback,
        #     10
        # )

        self.intrinsics_subscriber_
        self.pcd_subscriber_
        self.rgb_subscriber_

        self.get_logger().info("Camera-LiDAR calibration is initialized.")

        self.bridge = CvBridge()
        self.rsz_h = 192
        self.rsz_w = 640

        # default extrinsics and intrinsics
        self.R = np.array([[0.0431, -0.9990, -0.0116],
                           [-0.0883,  0.0078, -0.9961],
                           [0.9952,  0.0439, -0.0879]])
        self.t = np.array([[0.0], [0.05], [0.4]])
        self.T = np.vstack((np.hstack((self.R, self.t)), np.array([0., 0., 0., 1.])))
        
        self.K = np.array([[300., 0, 320.],
                           [0., 300., 240.],
                           [0., 0., 1.]])
        
        self.height = 720
        self.width = 1280

        self.pcd_proj = np.zeros((self.height, self.width))
        self.pcd_arr = np.zeros((1,3))

        # Model params
        config = {
                'model_name': 'TransCalib_LVT_EfficientNet_june2',
                'feature_matching' : {
                        'in_ch' : 512,  
                        'conv_repeat' : 3,
                        'conv_act' : 'elu',
                        'conv_drop' : 0.05,
                        'depthwise':[True, True, True],
                        'attn_repeat': [1, 1, 1],
                        'attn_types': ['csa', 'csa', 'csa'],
                        'attn_depths' : [2, 2, 2],
                        'embed_ch' : [1024, 1024, 2048],
                        'num_heads' : [4, 4, 8],
                        'mlp_ratios' : [4, 4, 4],  
                        'mlp_depconv' : [True, True, True], 
                        'attn_drop_rate' : 0.05, 
                        'drop_path_rate' : 0.05, 
                        },
                'regression_drop': 0.1
        }
        self.device = 'cuda'
        self.model = Model(DotWiz(config)).to(self.device)
        weight_path = "/home/wicomai/ros2_ws/src/etri_msfloc/etri_msfloc/calib_inference/TransCalib_LVT_EfficientNet_june2_20240606_074356_best_val.pth.tar"
        self.weight = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(self.weight["state_dict"])
        self.model.eval()

        self.rgb_transform = T.Normalize(mean=[0.33, 0.36, 0.33], 
                                           std=[0.30, 0.31, 0.32])
        self.depth_transform = T.Normalize(mean=[0.024439, 0.024439, 0.024439], 
                                           std=[0.12541, 0.12541, 0.12541])

    def intrinsics_callback(self, intr: CameraInfo):
        self.K = np.array(intr.k).reshape((3, 3))
        self.height = intr.height
        self.width = intr.width
        # self.get_logger().info(f'dim: ({self.height}, {self.width})')
        # self.get_logger().info(f'K: ({self.K})')

    def extrinsic_callback(self, extr: Extrinsics):
        self.R = np.array(extr.rotation).reshape((3,3))
        self.t = np.array(extr.translation).reshape((3,1))
        self.T = np.vstack((np.hstack((self.R, self.t)), np.array([0., 0., 0., 1.])))
        # self.T = np.linalg.inv(
        #                 np.vstack(np.hstack((self.R, self.t)), 
        #                           np.array([0., 0., 0., 1.])))
        # self.get_logger().info(f'T: ({self.T})')
    
    def pcd_callback(self, pcd: PointCloud2):
        self.pcd_arr = np.array(list(pc2.read_points(pcd, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
        self.get_logger().info(f'lidar point: {self.pcd_arr.shape}')

    def rgb_callback(self, img: Image):
        # self.get_logger().info('receiving image')

        # Convert ROS Image message to OpenCV image
        camera_img = self.bridge.imgmsg_to_cv2(img, "rgb8")
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_RGB2BGR)

        frame_rgb = cv2.resize(camera_img, (self.rsz_w, self.rsz_h))
        frame_rgb = frame_rgb/255.
        frame_rgb = torch.tensor(frame_rgb, dtype=torch.float32, device=self.device)
        frame_rgb = torch.unsqueeze(frame_rgb.permute(2,0,1), 0) # (H, W, C) --> (B, C, H, W)

        frame_depth, _, _, _ = self.project(self.pcd_arr[:,:3], self.T, self.K)
        frame_depth = cv2.resize(frame_depth, (self.rsz_w, self.rsz_h))
        frame_depth = np.array([frame_depth, frame_depth, frame_depth])
        frame_depth = frame_depth/255.
        frame_depth = torch.tensor(frame_depth, dtype=torch.float32, device=self.device)
        frame_depth = torch.unsqueeze(frame_depth, 0)

        rgb_in = self.rgb_transform(frame_rgb)
        depth_in = self.depth_transform(frame_depth)
        pcd_in =  torch.unsqueeze(torch.tensor(self.pcd_arr[:,:3], dtype=torch.float32, device=self.device), 0)
        T_in =  torch.unsqueeze(torch.tensor(self.T, dtype=torch.float32, device=self.device), 0)

        if(rgb_in.shape == depth_in.shape):

            _, T_new, _, _ = self.model(rgb_in, depth_in, pcd_in, T_in)

            self.T = torch.squeeze(T_new).detach().cpu().numpy()
            self.get_logger().info(f'getting prediction:{self.T}')

            _, u_proj, v_proj, d_proj = self.project(self.pcd_arr[:,:3], self.T, self.K)
            
            for u, v, d in zip(u_proj, v_proj, d_proj):
                if d<255/2:
                    color = (255-2*d, 2*d, 0)
                else:
                    color = (0, 510-2*d, 2*d-255)
                camera_img = cv2.circle(camera_img, (u,v), radius=0, color=color, thickness=4)
            
            camera_img = cv2.resize(camera_img, (int(self.width/2), int(self.height/2)))
            cv2.imshow("Calibration result", camera_img)
            cv2.waitKey(1)
        else:
            self.get_logger().info(f"different image size: {rgb_in.shape} and {depth_in.shape}")

    def project(self, point_cloud, T_ext, K_int):
        # rgb_img = img
        n_points = point_cloud.shape[0]
        pcd_cam = np.matmul(T_ext, np.hstack((point_cloud, np.ones((n_points, 1)))).T).T  # (P_velo -> P_cam)
        pcd_cam = pcd_cam[:,:3]
        z_axis = pcd_cam[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        pixel_proj = np.matmul(K_int, pcd_cam.T).T

        # normalize pixel coordinates
        pixel_proj = np.array([x/(x[2]+1e-6) for x in pixel_proj])

        u = np.array(pixel_proj[:, 0]).astype(np.int32)
        v = np.array(pixel_proj[:, 1]).astype(np.int32)

        # depth calculation of each point
        depth = z_axis #np.array([np.linalg.norm(x) for x in pcd_cam])

        condition = (0<=u)*(u<self.width)*(0<=v)*(v<self.height)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]

        out_img = np.zeros((self.height, self.width))
        if d_proj.shape[0] > 0:
            max_depth = np.max(d_proj)
            d_proj = np.array([np.interp(d, [0, max_depth], [255, 0]) for d in d_proj])
            out_img[v_proj,u_proj] = d_proj

        return out_img, u_proj, v_proj, d_proj
    
    def rot2qua(self, matrix):
        """
        Convert a rotation matrix to quaternion.
        Args:
            matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

        Returns:
            torch.Tensor: shape [4], normalized quaternion
        """
        if matrix.shape == (4, 4):
            R = matrix[:-1, :-1]
        elif matrix.shape == (3, 3):
            R = matrix
        else:
            raise TypeError("Not a valid rotation matrix")
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        q = np.zeros(4)
        if tr > 0.:
            S = math.sqrt(tr+1.0) * 2
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
        return q / np.linalg.norm(q)

    def rot2qua_torch(self, matrix):
        """
        Convert a rotation matrix to quaternion.
        Args:
            matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

        Returns:
            torch.Tensor: shape [4], normalized quaternion
        """
        if matrix.shape == (4, 4):
            R = matrix[:-1, :-1]
        elif matrix.shape == (3, 3):
            R = matrix
        else:
            raise TypeError("Not a valid rotation matrix")
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        q = torch.zeros(4, device=matrix.device)
        if tr > 0.:
            S = (tr+1.0).sqrt() * 2
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
        return q / q.norm()

    def qua2rot(self, q):
        """
        Convert a quaternion to a rotation matrix
        Args:
            q (torch.Tensor): shape [4], input quaternion

        Returns:
            torch.Tensor: [3x3] homogeneous rotation matrix
        """
        assert q.shape[0] == 4, "Not a valid quaternion"
        if np.linalg.norm(q) != 1.:
            q = q / np.linalg.norm(q)
        mat = np.zeros((3, 3))
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

    def qua2rot_torch(self, q):
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

def main(args=None):
    rclpy.init(args=args)
    calibrator = camera_lidar_calibration()
    rclpy.spin(calibrator)
    rclpy.shutdown()