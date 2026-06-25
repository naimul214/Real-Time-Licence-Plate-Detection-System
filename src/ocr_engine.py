import easyocr
import os
import time
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default save directory (configurable via env var, falls back to data/processed)
DEFAULT_SAVE_DIR = os.getenv(
    'OCR_SAVE_DIR', 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
)

def rename_image(image_path, detected_text, save_directory=DEFAULT_SAVE_DIR):
    """
    Renames the cropped license plate image file based on the detected text.

    Args:
        image_path (str): Current path to the cropped image.
        detected_text (str): The text extracted via OCR.
        save_directory (str): Target directory to save the renamed image.

    Returns:
        str: The filename of the renamed image, or the original filename if rename fails.
    """
    # Remove spaces and sanitize the text to form a clean filename
    license_plate = detected_text.replace(' ', '')
    clean_plate = re.sub(r'[^\w]', '', license_plate)
    if not clean_plate:
        clean_plate = "unknown"

    # Generate new filename with license plate and current timestamp
    timestamp = str(time.time()).replace('.', '')
    new_filename = f"{clean_plate}_{timestamp}.jpg"

    try:
        # Ensure target directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            logger.info(f"Created directory: {save_directory}")

        # Path to the new image file
        new_image_path = os.path.join(save_directory, new_filename)

        if os.path.exists(image_path):
            os.rename(image_path, new_image_path)
            logger.info(f"Renamed {os.path.basename(image_path)} -> {new_filename} inside {save_directory}")
            return new_filename
        else:
            logger.warning(f"Source image not found for renaming: {image_path}")
            return os.path.basename(image_path)

    except OSError as e:
        logger.error(f"Failed to rename image from {image_path} to {new_filename}: {e}")
        return os.path.basename(image_path)

def read_image(image_path, save_directory=DEFAULT_SAVE_DIR):
    """
    Opens an image file, reads the text using EasyOCR, and renames the file.

    Args:
        image_path (str): Path to the image containing the license plate.
        save_directory (str): Target directory to save the renamed image.

    Returns:
        str: The extracted license plate text.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist for OCR: {image_path}")
        return ""

    try:
        # Initialize EasyOCR reader (loads English models)
        reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available
        
        # Read text from the image
        text_data = reader.readtext(image_path)

        # Convert all detected text segments to a single string
        detected_text = ' '.join([result[1] for result in text_data]).strip()
        logger.info(f"Raw OCR Output: '{detected_text}'")

        # Remove specific regional phrases if they appear in the string
        phrases_to_remove = ['ONTARIO', 'ONTARIQ', 'YOURS TO DISCOVER', 'A PLACE TO GROW']
        for phrase in phrases_to_remove:
            detected_text = re.sub(re.escape(phrase), '', detected_text, flags=re.IGNORECASE)

        # Clean up double spaces resulting from removal
        detected_text = re.sub(r'\s+', ' ', detected_text).strip()
        logger.info(f"Filtered OCR Output: '{detected_text}'")

        # Rename the image file based on the detected text
        rename_image(image_path, detected_text, save_directory)

        return detected_text

    except Exception as e:
        logger.error(f"An error occurred during OCR processing: {e}", exc_info=True)
        return ""
