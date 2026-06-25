import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path to run directly if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.detector import yolo_predict, DEFAULT_MODEL_PATH
except ImportError:
    # If run in subfolders, try relative imports
    from ..src.detector import yolo_predict, DEFAULT_MODEL_PATH

def test_static_inference():
    """
    Runs end-to-end inference on the default sample test image to verify correctness.
    """
    image_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test.png')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    logger.info("Starting static image inference test...")
    logger.info(f"Using test image: {image_path}")
    logger.info(f"Using model weights: {DEFAULT_MODEL_PATH}")
    
    if not os.path.exists(image_path):
        logger.error(f"Test image not found at '{image_path}'. Cannot run verification.")
        sys.exit(1)
        
    # Run the prediction pipeline
    detected_text = yolo_predict(
        model_path=DEFAULT_MODEL_PATH,
        image_path=image_path,
        save_directory=output_dir
    )
    
    if detected_text is not None:
        logger.info("[SUCCESS] Pipeline executed successfully.")
        logger.info(f"Extracted License Plate Text: '{detected_text}'")
        print(f"\n==========================================")
        print(f"TEST RESULT: SUCCESS")
        print(f"Extracted License Plate: {detected_text}")
        print(f"==========================================\n")
    else:
        logger.error("[FAILURE] Inference executed, but no license plate was extracted or error occurred.")
        sys.exit(1)

if __name__ == '__main__':
    test_static_inference()
