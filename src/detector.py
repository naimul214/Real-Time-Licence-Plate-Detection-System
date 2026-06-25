import numpy as np
import cv2
from ultralytics import YOLO
import time
import os
import zipfile
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import OCR function with fallbacks for packaged/script execution
try:
    from src.ocr_engine import read_image
except ImportError:
    try:
        from ocr_engine import read_image
    except ImportError:
        # Fallback if pythonpath is not set up
        import sys
        sys.path.append(os.path.dirname(__file__))
        from ocr_engine import read_image

# Project root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best.pt')
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def check_best_model(model_path=DEFAULT_MODEL_PATH):
    """
    Checks if the YOLOv9 model weights exist. If not, attempts to unzip
    the included best.zip archive in the models folder.
    """
    models_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        logger.info(f"Model weight file '{model_path}' not found. Checking for zip archive...")
        zip_file_path = os.path.join(models_dir, 'best.zip')

        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(
                f"Neither '{os.path.basename(model_path)}' nor '{os.path.basename(zip_file_path)}' "
                f"was found in '{models_dir}'. Please provide model weights."
            )

        logger.info(f"Extracting weights from '{zip_file_path}'...")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            logger.info("Weights extracted successfully.")
        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Failed to unzip model weights: {e}")
            raise

def crop_image(results, image_path, save_directory=DEFAULT_PROCESSED_DIR):
    """
    Crops the image around the detected bounding box with the highest confidence.

    Args:
        results: Ultralytics YOLO detection results.
        image_path (str): Path to the source image.
        save_directory (str): Target directory to save the crop.

    Returns:
        str: Path to the saved crop image.
    """
    boxes = results[0].boxes.xyxy.tolist()
    confidences = results[0].boxes.conf.tolist()

    if not confidences:
        raise ValueError("No bounding boxes found in results.")

    # Select bounding box with the highest confidence score
    highest_conf_idx = np.argmax(confidences)
    highest_conf_box = boxes[highest_conf_idx]

    x1, y1, x2, y2 = highest_conf_box

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read source image: {image_path}")

    # Crop image based on bounding box coordinates
    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

    # Ensure save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save crop with timestamp to prevent collisions
    cropped_image_path = os.path.join(save_directory, f'crop_{time.time()}.jpg')
    cv2.imwrite(cropped_image_path, cropped_img)
    logger.info(f"Saved cropped plate image to: {cropped_image_path}")

    return cropped_image_path

def yolo_predict(model_path=DEFAULT_MODEL_PATH, image_path=None, save_directory=DEFAULT_PROCESSED_DIR):
    """
    Performs license plate detection and runs OCR on the highest-confidence crop.

    Args:
        model_path (str): Path to YOLOv9 weights.
        image_path (str): Path to input image.
        save_directory (str): Directory where cropped license plates are saved.

    Returns:
        str: Detected plate text or None.
    """
    if image_path is None or not os.path.exists(image_path):
        logger.error(f"Input image path does not exist: {image_path}")
        return None

    try:
        # Check and prepare model weights
        check_best_model(model_path)

        # Load model and run inference
        logger.info(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        logger.info(f"Running detection on: {image_path}")
        results = model(image_path)

        # Check if boxes were detected
        if len(results[0].boxes) == 0:
            logger.warning(f"No license plate detected in image: {image_path}")
            return None

        # Crop the detected region
        cropped_path = crop_image(results, image_path, save_directory)

        # Perform OCR
        plate_text = read_image(cropped_path, save_directory)
        logger.info(f"Successfully processed image. Detected Text: '{plate_text}'")
        return plate_text

    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv9 License Plate Detector and OCR CLI")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PROCESSED_DIR, help="Output folder for cropped plates")
    
    args = parser.parse_args()
    
    result = yolo_predict(args.model, args.image, args.output_dir)
    if result:
        print(f"\n[+] Extraction Successful!")
        print(f"License Plate Text: {result}\n")
    else:
        print(f"\n[-] Detection/OCR failed or no plate was found.\n")
