#!/usr/bin/env python
import rclpy
from rclpy.node import Node

class the_node(Node):
    def __init__(self):
        super().__init__("test_run")
        self.get_logger().info("Test 123")
        self.counter_ = 0
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info(f"k+{self.counter_}")
        self.counter_ += 1

def main(args=None):
    rclpy.init(args=args)
    
    node = the_node()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()