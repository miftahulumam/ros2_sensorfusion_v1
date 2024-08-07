import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from realsense2_camera_msgs.msg import Extrinsics
from .calib_inference import point_cloud2 as pc2

import numpy as np
import cv2
from cv_bridge import CvBridge

class pcd_projection(Node):
    def __init__(self):
        super().__init__("pcd_projection")

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

        self.intrinsics_subscriber_
        self.pcd_subscriber_
        self.rgb_subscriber_

        self.get_logger().info("Camera-LiDAR projection is initialized.")

        self.bridge = CvBridge()

        # default extrinsics and intrinsics
        self.R = np.array([[0.0431, -0.9990, -0.0116],
                           [-0.0883,  0.0078, -0.9961],
                           [0.9952,  0.0439, -0.0879]])
        self.t = np.array([[0.0], [0.2], [0.2]])
        self.T = np.vstack((np.hstack((self.R, self.t)), np.array([0., 0., 0., 1.])))
        
        self.K = np.array([[300., 0, 320.],
                           [0., 300., 240.],
                           [0., 0., 1.]])
        
        self.height = 720
        self.width = 1280

        self.pcd_proj = np.zeros((self.height, self.width))
        self.pcd_arr = np.zeros((1,3))

    def intrinsics_callback(self, intr: CameraInfo):
        self.K = np.array(intr.k).reshape((3, 3))
        self.height = intr.height
        self.width = intr.width
        # self.get_logger().info(f'dim: ({self.height}, {self.width})')
        # self.get_logger().info(f'K: ({self.K})')

    def pcd_callback(self, pcd: PointCloud2):
        self.pcd_arr = np.array(list(pc2.read_points(pcd, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
        self.get_logger().info(f'lidar point: {self.pcd_arr.shape}')

    def rgb_callback(self, img: Image):
        frame_rgb = self.bridge.imgmsg_to_cv2(img, "rgb8")
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        _, u_proj, v_proj, d_proj = self.project(self.pcd_arr[:,:3], self.T, self.K)

        for u, v, d in zip(u_proj, v_proj, d_proj):
            if d<255/2:
                color = (255-2*d, 2*d, 0)
            else:
                color = (0, 510-2*d, 2*d-255)
            frame_rgb = cv2.circle(frame_rgb, (u,v), radius=0, color=color, thickness=4)
            
        frame_rgb = cv2.resize(frame_rgb, (int(self.width/2), int(self.height/2)))
        cv2.imshow("Projection", frame_rgb)
        cv2.waitKey(1)

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

def main(args=None):
    rclpy.init(args=args)
    node = pcd_projection()
    rclpy.spin(node)
    rclpy.shutdown()