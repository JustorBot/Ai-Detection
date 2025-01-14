# 🚀 Object Detection Ai Model

An interactive Python-based application for object detection powered by the YOLO model. Detect objects in real-time using your webcam or analyze uploaded images with a highly customizable and user-friendly interface.

---

## 🌟 Features

### 🔍 **Webcam Integration**
- Real-time object detection with dynamic confidence thresholds.
- Customizable bounding boxes for better visibility.

### 🖼️ **Image Upload**
- Detect objects in uploaded images.
- Displays results in resizable windows with drawn bounding boxes.

### 🎨 **Custom Colors for 'Person' Class**
- Pick a custom color for the bounding boxes of objects detected as "person."

### 🎛️ **Confidence Threshold Adjustment**
- Use a slider to set the minimum confidence for displaying detections (0% - 100%).

### 🖥️ **Responsive User Interface**
- Dark-themed UI for enhanced visual comfort.
- Dynamically resizable result windows.
- Clear and user-friendly layout.

### 📈 **FPS Display**
- Real-time FPS display during live webcam detection.

---

## 🛠️ How to Run

### Prerequisites

1. **Install Anaconda**  
   Download and install [Anaconda](https://www.anaconda.com/products/distribution) (Python 3.7 or later).  

2. **Create a Virtual Environment**  
   Open the Anaconda prompt and run the following commands to create and activate a virtual environment:  
   ```bash
   conda create --name object-detection-env python=3.7
   conda activate object-detection-env

3. **Install Required Python Packages**
   Install the necessary dependencies in your virtual environment:
   ```bash
    pip install ultralytics pillow opencv-python tkinter
   
### Clone the Repository
  Clone the project to your local machine and navigate to the directory:
  ```bash
  git clone https://github.com/username/ObjectDetectionApp.git
  cd ObjectDetectionApp
```
### Run the Application
  Start the app with the following command:
  ```bash
  python object_detection_app.py
```

### Use the App Features
- Launch Camera: Start real-time object detection using your webcam.
- Upload Image: Choose an image file for object detection.
- Pick Person Color: Customize the bounding box color for "person" detections.
- Confidence Threshold Slider: Adjust the detection confidence dynamically.
- Exit: Close the application.
