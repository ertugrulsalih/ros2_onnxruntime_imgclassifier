import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor

import onnxruntime as ort
import numpy as np
import time

import cv2
from cv_bridge import CvBridge

# Options for TensorRT execution provider
trt_ep_options = {
    "trt_engine_cache_enable": True,  # Enable caching for TensorRT engine
    "trt_dump_ep_context_model": True,  # Dump engine context for debugging
    "trt_ep_context_file_path": "/home/ertugrul/ros2_ws/src/onnxruntime_classifier/models/trt_ep_context",  # Path to save context
}

# Set the inference provider to TensorRT with the options
providers = ['TensorrtExecutionProvider', trt_ep_options]

class OnnxRuntimeClassifier(Node):

    def __init__(self):
        super().__init__('onnxruntime_classifier_node')

        self.get_logger().info("init node: onnxruntime_classifier_node")

        self.publisher_ = self.create_publisher(String, 'inference_result', 10)

        self.subscription = self.create_subscription(Image, 'input_image', self.listener_callback, 10)

        start_times = time.time()

        # Load ONNX model using TensorRT provider
        self.ort_session = ort.InferenceSession(
            "/home/ertugrul/ros2_ws/src/ros2_onnxruntime_imgclassifier/models/onnx_model.onnx", #change this path
            providers=[("TensorrtExecutionProvider", trt_ep_options)]
        )
        time_elapsed = str(time.time() - start_times) + ' s'

        self.get_logger().info("ort Inference:  onnxruntime_classifier_node")
        self.get_logger().info(time_elapsed)

        self.bridge = CvBridge()

    def listener_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        resized_img = cv2.resize(cv_img, (256, 256))
        cropped_img = resized_img[16:240, 16:240]

        # Preprocess the image for the ONNX model (transpose and normalize)
        img_t = np.transpose(cropped_img, (2, 0, 1)) / 255.0
        img_t = np.expand_dims(img_t, axis=0).astype(np.float32)

        # Measure the time taken for inference
        start_time = time.time()
        ort_inputs = {self.ort_session.get_inputs()[0].name: img_t}
        ort_outs = self.ort_session.run(None, ort_inputs)
        end_time = time.time()
        avg_time = (end_time - start_time)

        result_class = np.argmax(ort_outs)
        
        # Create a result string with the inference time and predicted class
        result_str = f"Inference Time (TensorRT Provider): {(avg_time * 1000):.6f} ms, Class:{result_class}"

        self.publisher_.publish(String(data=result_str))
        
        self.get_logger().info(result_str)

def main(args=None):
    rclpy.init(args=args)
    node = OnnxRuntimeClassifier()

    # Use a MultiThreadedExecutor to allow for multi-threaded processing
    executor = MultiThreadedExecutor(num_threads=1)

    executor.add_node(node)
    try:
        executor.spin()  # Keep the node running and processing callbacks
    finally:
        node.destroy_node()  
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
