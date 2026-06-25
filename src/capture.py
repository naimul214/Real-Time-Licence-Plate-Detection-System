import cv2
import os
import time
import logging
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import detector with fallbacks for packaged/script execution
try:
    from src.detector import yolo_predict, DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_DIR
except ImportError:
    try:
        from detector import yolo_predict, DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_DIR
    except ImportError:
        import os
        sys.path.append(os.path.dirname(__file__))
        from detector import yolo_predict, DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_DIR

# Define default paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_LOG_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'license_plates.txt')

def take_snapshot(camera_source=0, save_directory=DEFAULT_PROCESSED_DIR):
    """
    Captures a single frame from the camera source, displays it, and saves it.

    Args:
        camera_source: Integer for local webcam index, or String for RTSP/file stream.
        save_directory (str): Directory where the captured frame is saved.

    Returns:
        str: Absolute path to the saved frame, or None if capture fails.
    """
    cam = None
    try:
        logger.info(f"Opening camera source: {camera_source}")
        # Resolve camera source type
        if isinstance(camera_source, str) and camera_source.isdigit():
            camera_source = int(camera_source)

        cam = cv2.VideoCapture(camera_source)
        if not cam.isOpened():
            logger.error(f"Could not open camera source: {camera_source}")
            return None

        # Capture a frame
        ret, frame = cam.read()
        if not ret:
            logger.error("Failed to read frame from camera.")
            return None

        # Display the frame briefly
        cv2.imshow("Webcam License Plate Detection", frame)
        key = cv2.waitKey(2000)  # Show for 2 seconds (non-blocking for UI update)
        cv2.destroyAllWindows()

        if key == 27:  # ESC key pressed
            logger.info("Escape key pressed. Exiting capture.")
            return None

        # Ensure directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save snapshot
        filename = os.path.join(save_directory, f'snapshot_{round(time.time())}.jpg')
        if cv2.imwrite(filename, frame):
            logger.info(f"Snapshot saved to: {filename}")
            return filename
        else:
            logger.error(f"Failed to write snapshot to disk: {filename}")
            return None

    except Exception as e:
        logger.error(f"An error occurred during snapshot capture: {e}", exc_info=True)
        return None
    finally:
        if cam is not None:
            cam.release()
            cv2.destroyAllWindows()

def cam_predict(model_path=DEFAULT_MODEL_PATH, camera_source=0, 
                save_directory=DEFAULT_PROCESSED_DIR, log_file=DEFAULT_LOG_FILE):
    """
    Takes a webcam snapshot, runs detection + OCR, and logs the result to a text file.

    Args:
        model_path (str): Path to YOLO weights.
        camera_source: Webcam index or IP camera stream URL.
        save_directory (str): Directory to save cropped plate images.
        log_file (str): Log file to append detected plates.

    Returns:
        str: Detected plate text or None.
    """
    image_path = take_snapshot(camera_source, save_directory)
    if image_path is None:
        logger.warning("Camera snapshot could not be captured. Skipping this frame.")
        return None

    # Perform YOLO prediction and OCR
    license_plate = yolo_predict(model_path, image_path, save_directory)

    if license_plate:
        try:
            # Create parent folder for log file if missing
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            with open(log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {license_plate}\n")
            
            logger.info(f"SAVED: '{license_plate}' logged to {log_file}")
            return license_plate
        except OSError as e:
            logger.error(f"Failed to write license plate to log file {log_file}: {e}")
            return license_plate
    else:
        logger.info("No license plate detected in snapshot.")
        return None

def plate_scanner(model_path=DEFAULT_MODEL_PATH, camera_source=0, 
                  save_directory=DEFAULT_PROCESSED_DIR, log_file=DEFAULT_LOG_FILE, interval=15):
    """
    Runs a continuous loop taking snapshots and detecting plates at specified intervals.
    """
    logger.info(f"Starting Plate Scanner Loop. Interval: {interval}s. Target Log: {log_file}")
    logger.info("Press Ctrl+C to terminate the loop.")
    
    try:
        while True:
            cam_predict(model_path, camera_source, save_directory, log_file)
            logger.info(f"Sleeping for {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Plate Scanner loop stopped by user request.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Webcam License Plate Scanner CLI")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLO weights")
    parser.add_argument('--camera', default=0, help="Webcam index (int) or RTSP Stream URL (str)")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PROCESSED_DIR, help="Directory to save image crops")
    parser.add_argument('--log-file', type=str, default=DEFAULT_LOG_FILE, help="File to append extracted plate logs")
    parser.add_argument('--interval', type=int, default=15, help="Interval in seconds between frames")
    
    args = parser.parse_args()
    
    plate_scanner(
        model_path=args.model,
        camera_source=args.camera,
        save_directory=args.output_dir,
        log_file=args.log_file,
        interval=args.interval
    )
