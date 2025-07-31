#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from self_driving_car_pkg.Drive_Bot import Car
import cv2
import numpy as np

class ComputerVisionNode(Node):
    def __init__(self):
        super().__init__('computer_vision_node')
        self.get_logger().info("🚗 Starting Autonomous Car - Computer Vision Node 🚗")
        
        # Verify OpenCV installation
        test_img = np.zeros((100,100,3), dtype=np.uint8)
        cv2.imshow("System Check", test_img)
        cv2.waitKey(1)
        
        self.bridge = CvBridge()
        self.car = Car(side='right')
        self.get_logger().info("Car controller initialized")
        
        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Subscribed to camera feed")
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Command publisher ready")
        
        # Emergency motion counter
        self.stuck_counter = 0
        self.debug = True
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if frame is None or frame.size == 0:
                self.get_logger().warn("⚠️ Empty frame received!")
                return
            
            # Process frame
            processed_img, speed, angle = self.car.driveCar(frame)
            
            # Emergency motion if stuck
            if speed == 0 and angle == 0 and self.car.control.current_confidence == 0:
                self.stuck_counter += 1
                if self.stuck_counter > 15:  # After 15 frames of no movement
                    cmd = Twist()
                    cmd.linear.x = 0.5  # Strong push forward
                    self.cmd_pub.publish(cmd)
                    self.get_logger().warn("🆘 Applying emergency motion!")
                    return
            else:
                self.stuck_counter = 0
            
            # Publish normal command
            cmd = Twist()
            cmd.linear.x = float(speed)
            cmd.angular.z = float(angle)
            self.cmd_pub.publish(cmd)
            
            # Debug output
            if self.debug:
                cv2.imshow('Lane Detection', processed_img)
                cv2.waitKey(1)
                
                self.get_logger().info(
                    f"CONTROL: Speed={speed:.2f}m/s | "
                    f"Angle={angle:.2f}rad | "
                    f"Confidence={self.car.control.current_confidence:.2f}"
                )
                
        except Exception as e:
            self.get_logger().error(f"❌ Error: {str(e)}")
            # Emergency stop
            cmd = Twist()
            self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ComputerVisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutting down...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()