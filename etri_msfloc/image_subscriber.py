#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class RGB_image_subscriber(Node):
    def __init__(self):
        super().__init__('get_rgb_image')
        self.subscribe = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.img_callback,
            10
        )

    def img_callback(self, msg: Image):
        self.get_logger().info(msg.header)

def main(args=None):
    rclpy.init(args=args)

    img_subscribe = RGB_image_subscriber()
    rclpy.spin(img_subscribe)

    rclpy.shutdown()

if __name__ == '__main__':
    main()


    