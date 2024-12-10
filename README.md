
# NVIDIA GPU Accelerated Object Detection with YOLO and OpenCV

## **Synopsis**
This project demonstrates real-time object detection using the **YOLOv3** model integrated with **OpenCV** and accelerated by **CUDA** on an **NVIDIA GPU**. The goal is to leverage GPU processing to achieve high-speed, efficient detection of objects in images and video streams. The project provides a foundational implementation that can be extended to various real-world applications such as surveillance, traffic monitoring, and smart automation.

---

## **Key Features**
- **Object Detection with YOLO:** Utilizes the YOLOv3 model trained on the COCO dataset to identify 80 object categories.
- **GPU Acceleration:** Achieves high performance using NVIDIA CUDA for parallel computation.
- **Flexible Input Support:** Supports image files, video streams, and live camera feeds.
- **Customizable Detection:** Adjustable confidence thresholds and detection parameters.
- **Extensible:** Modular design for easy integration into larger projects.

---

## **Technologies Used**
- **YOLOv3:** Pre-trained deep learning model for fast object detection.
- **OpenCV:** For image processing, video capture, and drawing results.
- **CUDA and NVIDIA GPUs:** For accelerating the object detection process.
- **Python:** Programming language used to integrate all components.

---

## **Setup and Installation**
### **Prerequisites**
1. **Python 3.x** installed on your system.
2. **NVIDIA GPU** with CUDA support and the CUDA toolkit installed.
3. Required libraries:
   - OpenCV (`opencv-python`, `opencv-contrib-python`)
   - NumPy
   - Matplotlib (optional, for visualizations)

### **Installation Steps**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/nvidia-yolo-opencv.git
   cd nvidia-yolo-opencv
   ```

2. Install the required Python libraries:
   ```bash
   pip install opencv-python opencv-contrib-python numpy
   ```

3. Download YOLOv3 files:
   - `yolov3.weights`: [Download Here](https://pjreddie.com/media/files/yolov3.weights)
   - `yolov3.cfg`: [Download Here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - `coco.names`: [Download Here](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
   Save these files in the `model` directory.

---

## **Usage**
1. **Run the Object Detection Script**
   - Replace `input_image.jpg` with your image file.
   ```bash
   python object_detection.py --image input_image.jpg
   ```

2. **Real-Time Detection with Webcam**
   ```bash
   python object_detection.py --webcam
   ```

3. **Video Input**
   - Replace `input_video.mp4` with your video file.
   ```bash
   python object_detection.py --video input_video.mp4
   ```

---

## **Configuration**
You can adjust the following parameters in the `object_detection.py` file:
- **Confidence Threshold:**
  Controls the minimum confidence for object detection.
  ```python
  confidence_threshold = 0.5
  ```
- **NMS Threshold:**
  Adjusts the non-max suppression threshold.
  ```python
  nms_threshold = 0.4
  ```

---

## **Example Output**
### **Input Image**
![Input Image](example_input.jpg)

### **Detected Objects**
![Detected Output](example_output.jpg)

---

## **License**
This project is open-source and licensed under the MIT License. Feel free to use, modify, and distribute.

---

## **Contributions**
Contributions are welcome! If you'd like to improve this project or add new features:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## **Contact**
For any queries, suggestions, or collaborations, feel free to contact:
- **Name:** [Nagamohan Kumar Palakurthy]
- **Email:** nagaCodesAI@gmail.com
- **GitHub:** [Your GitHub Profile](https://github.com/nagacodesai)
