
# **NVIDIA GPU Accelerated Object Detection with YOLO and OpenCV**

## **Synopsis**

This project demonstrates real-time object detection using the **YOLOv8** model integrated with **OpenCV**. The system automatically manages YOLO weights, downloads them if necessary, and performs object detection on input images. The results are saved and displayed in a user-friendly, resizable window. The project is GPU-accelerated using NVIDIA's CUDA for high performance.
Work in Progress  - ETA end of Dec, 2024
---

## **Key Features**
- **Automatic Weight Management:**
  - Downloads YOLOv8 weights if not available locally and organizes them in a structured directory.
- **Object Detection:**
  - Uses the YOLOv8 model for object detection in images.
- **Friendly Image Display:**
  - Annotated results are saved and displayed in a resizable OpenCV window.
- **Modular Design:**
  - Reusable functions for weight management and image processing.

---

## **Technologies Used**
- **YOLOv8:** A state-of-the-art object detection model by Ultralytics.
- **OpenCV:** For image processing and result display.
- **Python:** Programming language used to integrate all components.

---

## **Setup and Installation**
### **Prerequisites**
1. **Python 3.8 or later** installed on your system.
2. **NVIDIA GPU** with CUDA support and the CUDA toolkit installed.
3. Required Python libraries:
   - Ultralytics YOLO
   - OpenCV
   - NumPy

## **What is YOLO?**
YOLO (You Only Look Once) is a real-time object detection algorithm that performs detection in a single pass through the neural network. Known for its balance between speed and accuracy, YOLO is widely used in applications like autonomous vehicles, surveillance, and robotics.

---

## **What is YOLOv8?**
YOLOv8 is the latest version developed by **Ultralytics** and released in 2023. It extends the capabilities of the YOLO family by supporting multiple tasks like object detection, instance segmentation, and pose estimation. Designed for usability and deployment versatility, YOLOv8 offers seamless integration with various deployment platforms.

| **Feature**        | **Details**                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Developer**       | Ultralytics                                                                |
| **Release Year**    | 2023                                                                       |
| **Key Advancements**| Supports object detection, segmentation, and pose estimation.              |
| **Export Options**  | Easily exportable to ONNX, CoreML, TensorRT, and other formats.            |
| **Flexibility**     | Optimized for real-time applications, edge devices, and cloud deployments. |

---

## **YOLOv8 Variants**
YOLOv8 offers multiple variants to cater to different performance and hardware requirements:

| **Variant**       | **Purpose**                                                                                  | **Description**                                                                                                   |
|-------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **YOLOv8n**       | Nano                                                                                         | Smallest and fastest variant, designed for low-resource environments and edge devices.                           |
| **YOLOv8s**       | Small                                                                                        | Balanced between speed and accuracy, suitable for real-time applications on moderate hardware.                   |
| **YOLOv8m**       | Medium                                                                                       | Better accuracy while maintaining reasonable speed. Ideal for general-purpose tasks.                             |
| **YOLOv8l**       | Large                                                                                        | High accuracy, suitable for applications where performance is prioritized over speed.                            |
| **YOLOv8x**       | Extra Large                                                                                  | Largest model in the family, providing the best accuracy for demanding tasks with sufficient computational power. |

---

## **Purpose of YOLOv8**
YOLOv8 is versatile and designed for a variety of applications:

| **Application**               | **Details**                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------|
| **Object Detection**           | Identifies objects in an image or video and draws bounding boxes around them.                       |
| **Instance Segmentation**      | Detects objects and masks their shapes in images or videos.                                         |
| **Pose Estimation**            | Estimates human or object poses by identifying key points like joints or corners.                   |
| **Edge Computing**             | Variants like YOLOv8n are optimized for resource-constrained environments such as IoT devices.      |
| **Cloud and Server Deployment**| Supports large-scale deployments with high-performance models like YOLOv8l and YOLOv8x.             |

---

## **Advantages of YOLOv8**
- **Versatility:** Supports detection, segmentation, and pose estimation tasks in one unified model.
- **Ease of Use:** Includes pre-trained models and export capabilities for deployment to ONNX, CoreML, TensorRT, and more.
- **Performance Scaling:** Offers a range of variants to balance speed and accuracy for different hardware environments.


### **Installation Steps**
1. Clone this repository:
   ```bash
   git clone https://github.com/nagacodesai/NVIDIA-GPU-Accelerated-Object-Detection-with-YOLO-and-OpenCV.git
   cd NVIDIA-GPU-Accelerated-Object-Detection-with-YOLO-and-OpenCV
   ```

2. Install the required Python libraries:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

---

## **Usage**
1. **Run the Script**
   - Replace `IMG_2496.jpeg` with your input image file:
     ```bash
     python detect_image.py
     ```

2. **Process Workflow:**
   - **Weights Management:** The script checks for the existence of YOLOv8 weights and downloads them if unavailable.
   - **Object Detection:** The script detects objects in the input image and saves an annotated version.
   - **Display Results:** The annotated image is displayed in a resizable OpenCV window.

---

## **Example Output**
### **Input Image**
![Input Image](example_input.jpg)

### **Detected Objects**
![Detected Output](example_output.jpg)

---

## **Functions Overview**
### **1. `checkWeightsReturnModel()`**
- Checks for existing YOLO weights or downloads them if missing.
- Ensures the weights are organized in the designated directory.
- Determine device: Use CUDA if available, otherwise CPU.device = 'cuda' if torch.cuda.is_available() else 'cpu'
- Returns the YOLO model.

### **2. `detectImage(imgFile, model)`**
- Detects objects in the given image using the YOLO model.
- Saves the annotated image with bounding boxes and labels.
- Displays the annotated image in a resizable OpenCV window.

---

## **Configuration**
- **Input Image:** Modify the `imgFle` variable in the script to change the input image.
- **Weights Path:** Update the `weights_dir` variable to change the directory for storing YOLO weights.

---

## **License**
This project is open-source and licensed under the MIT License. Feel free to use, modify, and distribute.

---


---

## **Contact**
For any queries, suggestions, or collaborations, feel free to contact:
- **Name:** Nagamohan Kumar Palakurthy
- **Email:** nagaCodesAI@gmail.com
- **GitHub:** [Nagamohan Kumar Palakurthy](https://github.com/nagacodesai)
