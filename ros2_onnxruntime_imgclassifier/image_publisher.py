import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_=self.create_publisher(Image, 'input_image', 10)
        self.timer = self.create_timer(0.04, self.timer_callback)

        self.bridge = CvBridge()

    def timer_callback(self):
        img = cv2.imread('/home/ertugrul/ros2_ws/src/ros2_onnxruntime_imgclassifier/test_images/Cat_November_2010-1a.jpg')
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')

        self.publisher_.publish(img_msg)

        self.get_logger().info('Image Sent')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()