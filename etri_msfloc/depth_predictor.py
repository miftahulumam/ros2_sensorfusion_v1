#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

import torch
from .depth_inference.model.net import Net


class depth_predictor(Node):
    def __init__(self):
        super().__init__("depth_prediction")
        self.depth_publisher_ = self.create_publisher(
            Image,
            "/depth_prediction",
            10
        )
        self.rgb_subscriber_ = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.rgb_callback,
            10
        )
        self.rgb_subscriber_
        self.get_logger().info("Depth prediction is initialized.")

        self.bridge = CvBridge()
        self.rsz_h = 192
        self.rsz_w = 640

        self.device = 'cuda'
        self.weight = '/home/wicomai/ros2_ws/src/etri_msfloc/etri_msfloc/depth_inference/best_model/best_val_loss.pth'
        self.model = Net('MonoUp')
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.weight, map_location=self.device))
        self.model.eval()

    def rgb_callback(self, img: Image):
        self.get_logger().info('receiving image')

        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(img, "rgb8")

        frame = cv2.resize(frame, (self.rsz_w, self.rsz_h))
        # frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.
        frame = torch.tensor(frame, dtype=torch.float32, device=self.device)
        frame = frame.permute(2,0,1)
        frame = torch.unsqueeze(frame, 0)
        out = self.model(frame, frame)
        output = out[0][0][0].detach().cpu().numpy()
        self.get_logger().info(f'getting prediction:{output.dtype}')
        plt.imshow(output, cmap='magma')
        plt.pause(.8)
        # out_frame = self.bridge.cv2_to_imgmsg(output, 'mono8')
        # out_frame = self.bridge.cv2_to_imgmsg(frame_grayscale, 'mono8')

        # self.depth_publisher_.publish(out_frame)
        self.get_logger().info('Img published')


def main(args=None):
    rclpy.init(args=args)
    predictor = depth_predictor()
    rclpy.spin(predictor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()