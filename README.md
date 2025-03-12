# License Plate Detection and OCR

This project performs automatic license plate detection using a YOLOv9 model and extracts text from the detected plate using Optical Character Recognition (OCR).

## Features
- Uses YOLOv9 for license plate detection.
- Crops the detected license plate with the highest confidence.
- Applies OCR to extract text from the cropped plate image.
- Automatically extracts the YOLO model (`best.pt`) if not found.

## Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- pip
- virtualenv (optional but recommended)

### Setup
```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Running the Detection and OCR
```python
from yolo_ocr import yolo_predict

model_path = "best.pt"
image_path = "path/to/image.jpg"

license_plate_text = yolo_predict(model_path, image_path)
print("Detected License Plate:", license_plate_text)
```

### 2. Directory Structure
```
project_root/
│── cropped_plates/   # Stores cropped license plate images
│── best.pt           # YOLOv9 trained model (automatically extracted if missing)
│── best.zip          # Zip file containing best.pt
│── yolo_ocr.py       # Main script for YOLO detection and OCR
│── ocr.py            # OCR script
│── requirements.txt  # Required dependencies
│── README.md         # Project documentation
```

## Model Check and Extraction
If the `best.pt` model file is missing, the script will automatically attempt to extract it from `best.zip`. If neither is found, the script will prompt you to provide the model file.

## Dependencies
- `numpy`
- `opencv-python`
- `ultralytics` (for YOLOv9)
- `pytesseract` (for OCR, ensure Tesseract-OCR is installed)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Notes
- Ensure that `best.pt` is present in the root directory or inside `best.zip`.
- The script will save cropped license plate images inside `cropped_plates/`.
- You may need to configure `pytesseract` by specifying the path to `tesseract.exe` if using Windows.

## License
MIT License

