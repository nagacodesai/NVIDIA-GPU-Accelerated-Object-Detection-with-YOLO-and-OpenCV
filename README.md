
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
