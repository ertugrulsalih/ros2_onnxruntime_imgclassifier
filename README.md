# ROS2 Inference with ONNX and TensorRT

## Overview

The project aimed to run an optimised model with TensorRT in ROS 2 using ONNX Runtime. It processed an image every 0.04 seconds and obtained the classification result. It includes two key nodes:

- `onnxruntime_classifier_node:` This node performs inference on incoming images and publishes the result, including the inference time and the predicted class.
- `image_publisher_node:` This node publishes an image from a local directory to the ROS2 topic for inference.
  
The project was developed to explore how artificial intelligence models can be integrated and utilized in ROS2 systems, specifically using ONNX Runtime with TensorRT for accelerated inference.

The inference node I developed using the ONNX Runtime runs using an optimised ONNX model with the TensorRT provider. 
In this node, using the publisher-subscriber mechanism of ROS 2, an image is received and processed every 0.04 seconds and the inference time is calculated with the classification result. 
One of the most important points here is the process of managing the dependencies of the model.
In models converted to ONNX format, the dependencies used during training do not need to be added again during inference. This increases the portability of the model and makes it easier to integrate into an environment such as ROS 2. 
That is, libraries used in the training phase (e.g. PyTorch or TensorFlow) do not have to be used by the ROS 2 node in the inference phase of the model. 
This makes the inference node more lightweight and free of dependencies. The dependencies of the ONNX model used during this work are the following:

- **ONNX Runtime:** The main engine for running the model.
- **TensorRT:** Provider used to optimise the model and provide acceleration on the GPU.

When we integrated with ROS 2, we made sure that only the dependencies needed for inference were used. For example, dependencies such as PyTorch used to save the trained model didn't need to be added to the ROS 2 node. 
Once the model is saved, it becomes fully portable in .onnx format and can work independently of the dependencies used in the training process during inference.

## Package Details

**TensorRT Engine Caching**
In this project, TensorRT engine caching is enabled with the following options:
```python
trt_ep_options = {
    "trt_engine_cache_enable": True,
    "trt_dump_ep_context_model": True,
    "trt_ep_context_file_path": "/path/to/cache"
}
```
***Why Use TensorRT Engine Cache?***
- **Purpose:** The TensorRT engine cache stores the optimized inference engine created from the ONNX model. When the model is first loaded, TensorRT optimizes the network for the target hardware (e.g., GPU). This process can take time, but with caching, the optimized engine is saved and reused in subsequent runs, reducing the startup time for inference.
- **Benefit:** By enabling the cache, you avoid the repeated cost of model optimization, improving the overall performance of the system, especially in scenarios where the model is loaded multiple times.
- **Debugging:** The trt_dump_ep_context_model option allows saving the TensorRT engine context, which is useful for debugging and analyzing the model's performance.

***Why Use MultiThreadedExecutor?***
- **Purpose:** ROS2 uses executors to manage the execution of tasks like subscribing to topics, publishing messages, or responding to service requests. The MultiThreadedExecutor allows multiple tasks to be handled in parallel using multiple threads.
- **Benefit:** This is particularly useful in systems where nodes need to handle multiple callbacks simultaneously, such as processing incoming images and publishing results. With multi-threading, the executor can improve performance by processing different tasks concurrently, leading to more efficient use of system resources.
- **Use Case in the Project:** In this project, using the MultiThreadedExecutor ensures that the image subscription and inference operations do not block each other, allowing smoother real-time performance for inference and result publishing.

## Test Configuration and Results (GTX1650ti Laptop Ubuntu 22.04)

| **package**     | **Version** |
|-----------------|-------------|
| onnxruntime-gpu |    1.19.0   |
| tensorrt        |    10.2.0   |
| numpy           |    1.26.3   |
| openCV          |  4.10.0.84  |
| CUDA            |     12.4    |
| cuDNN           |     9.4     |
| ROS2            |    Humble   |


**Results:**
| **Provider**      | **ROS2 OnnxRuntime inference speed (ms)** |
|-------------------|-------------------------------------------|
| CPU Provider      |                    ~35                    |
| CUDA Provider     |                     ~9                    |
| TensorRT Provider |                     ~5                    |

## Installation and Usage Package
### Installation 

**1 -** Clone the repository to your ROS2 workspace:

```bash
git clone <repo-url>
```

**2 -** Install the required dependencies for ONNX Runtime and OpenCV:

```bash
pip install onnxruntime-gpu==1.19.0 opencv-python cv_bridge numpy==1.26.3
```

**3 -** Ensure TensorRT is installed on your system and properly configured. You can refer to NVIDIA's [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1020/index.html) for more details. [For Onnxruntime TensorRT Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

**4 -** Build the ROS2 package:

**5 -** Source the workspace:

### Usage

**1 -** Running the Inference Node
The onnxruntime_classifier_node listens to the input_image topic and processes incoming images. To start this node:

```bash
ros2 run onnxruntime_classifier inference_node
```

**2 -** Running the Image Publisher Node
To publish an image for inference, use the image_publisher_node. This node reads an image file and publishes it to the input_image topic:

```bash
ros2 run onnxruntime_classifier publisher_node
```

Make sure you have a valid image in the test_images/ folder or update the path in the code to point to a different image.

**3 -** Inference Output
Once the inference_node is running and receiving images, it will log the following information:
- Inference Time: Time taken to process the image (in milliseconds).
- Class Prediction: The class label predicted by the ONNX model.

You should see output like this in the terminal:

```less
[INFO] [inference_node]: Inference Time (TensorRT Provider): 10.512 ms, Class: 282
```

## Conclusion

- **Dependencies:**

When working with ONNX Runtime, there is no need to add the dependencies of the trained model again during inference. This is especially true with the portability of the .onnx format and the ability of the model to run independently of the dependencies in the training phase. However, it is important to consider that the model needs the libraries used for inference. That is, dependencies such as ONNX Runtime, TensorRT, etc. may be required during inference, but PyTorch or other libraries used during model training will not be required during inference.

- **ONNX Runtime and TensorRT:**

Accelerating ONNX models using the TensorRT provider provides a significant performance advantage in the inference process. TensorRT enables faster inference on the GPU and is suitable for use in ROS 2.
Additional Information: While TensorRT provides performance optimisation on the GPU, very small performance differences may occur for some model types. For example, for low complexity models, the FPS difference between CUDA and TensorRT may not be very large (in our case there is a small difference between CUDA and TensorRT). This depends on the model and hardware configuration and it is normal that the expected improvement in each scenario is not large.

- **Implementation with C++:**

Writing a ROS 2 node using the ONNX Runtime C++ API is possible and can potentially provide better performance. However, there may be some differences in management between the Python API and the C++ API, and C++ requires a more complex development process. Therefore, using the C++ API may be preferable, especially in projects with high performance requirements.

- **ROS 2 Integration:**

It is a correct method to integrate the model with ROS 2 and publish the results by making time measurements. Your approach of taking an image every 0.04 seconds and performing classification is suitable for a practical real-time system. However, it may be useful to check whether performance is affected when broadcasting too frequently on ROS 2 nodes (in terms of message queuing and processing queues).

