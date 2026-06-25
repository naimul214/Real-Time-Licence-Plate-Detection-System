# Real-Time License Plate Detection & OCR System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv9](https://img.shields.io/badge/YOLOv9-ultralytics-orange)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Inference-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A production-ready pipeline that detects vehicle license plates in real-time using a YOLOv9 object detection model and extracts plate numbers using EasyOCR.

## The "Why" (Real-World Value)
Automated License Plate Recognition (ALPR) is a core component in modern smart cities, parking management, and security systems. This project automates the detection and extraction of license plate numbers from video streams or static images, reducing manual entry overhead, improving traffic log accuracy, and enabling automated access control at entry gates.

## Tech Stack
* **Object Detection:** YOLOv9 (via Ultralytics)
* **Optical Character Recognition (OCR):** EasyOCR (PyTorch-based)
* **Image Processing & Capture:** OpenCV
* **Containerization:** Docker
* **Core Language:** Python 3.10

## System Workflow & Architecture
```
[ Video Stream / Camera ] ──> [ Frame Capture (OpenCV) ] 
                                     │
                                     ▼
                      [ License Plate Detection (YOLOv9) ]
                                     │
                                     ▼
                       [ High-Confidence Bounding Box ]
                                     │
                                     ▼
                        [ Image Cropping & Filtering ]
                                     │
                                     ▼
                         [ OCR Extraction (EasyOCR) ]
                                     │
                                     ▼
                    [ Clean Text Output & Saved Image ]
```

## Quickstart Guide

### Prerequisites
* Python 3.10+
* Camera input (if running real-time webcam capture)
* Docker (optional, for containerized run)

### 1. Local Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/naimul214/Real-Time-Licence-Plate-Detection-System.git
cd Real-Time-Licence-Plate-Detection-System

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Inference
The system automatically extracts the trained model weights (`best.pt`) from the included `best.zip` archive on first run.

* **Run on a static test image:**
  ```bash
  python src/detector.py --image data/raw/test.png
  ```

* **Run real-time webcam capture loop:**
  ```bash
  python src/capture.py --camera 0
  ```

### 3. Run with Docker (Containerized)
To build and run the application in a container:
```bash
docker build -t license-plate-detector .
docker run --device=/dev/video0 license-plate-detector
```

## Results & Performance
* **Detection Confidence:** The fine-tuned YOLOv9 model detects license plates under varying lighting conditions with high confidence.
* **Inference Speed:** Average processing speed of ~XX ms per frame on [Insert Hardware, e.g., NVIDIA Jetson Nano / Apple M1 / CPU].
* **OCR Accuracy:** EasyOCR achieves high accuracy on clean crops, with built-in regex-based filtering to strip out regional plate noise (e.g., "Ontario").

*(Placeholder: Insert a demo GIF or screenshot showing the bounding box overlay and OCR output text)*

## Limitations & Future Work
* **Angle & Perspective Distortion:** Extremely oblique angles reduce OCR reliability; adding a spatial transformer network (STN) or perspective correction step would improve performance.
* **Temporal Consistency:** Currently processes frames independently; introducing a multi-object tracker (e.g., ByteTrack) would prevent duplicate detections and smooth plate text across frames.
* **Hardware Optimization:** TensorRT optimization is needed to deploy this on low-power edge hardware at maximum frame rates.
* **Database Integration:** Future iterations will pipe extracted license numbers and timestamps directly into a PostgreSQL/SQLite database or cloud logging service.

## Connect
* **GitHub:** [naimul214](https://github.com/naimul214)
* **LinkedIn:** [Naimul](https://linkedin.com/in/naimul214)
