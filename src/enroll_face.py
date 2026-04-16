# enroll_face.py
import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple

# --- Setup Logging First ---
try:
    from src.log_config import setup_logging
    setup_logging()
except ImportError:
    # Basic fallback if log_config is not found
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.warning("src.log_config not found, using basic logging.")

# --- Import project modules AFTER logging is set up ---
from src.detection import FaceDetector
from src.embedding import FaceEmbedder
from src.database import EmbeddingDatabase
from src import config # Import configuration
from src.utils import check_image_quality

logger = logging.getLogger(__name__)

def select_best_face(bboxes: np.ndarray, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Selects the best face candidate for enrollment (e.g., largest, most central).

    Args:
        bboxes: Array of bounding boxes from detector.
        landmarks: Array of landmarks from detector.
        frame_shape: Tuple (height, width) of the original frame.

    Returns:
        A tuple (selected_bbox, selected_landmark) or None if no suitable face found.
    """
    if bboxes is None or landmarks is None or bboxes.shape[0] == 0:
        return None

    best_face_idx = -1
    frame_center_x = frame_shape[1] / 2
    frame_center_y = frame_shape[0] / 2

    candidates = []

    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2, score = bboxes[i]
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Filter 1: Minimum size
        if w < config.MIN_FACE_SIZE_ENROLL or h < config.MIN_FACE_SIZE_ENROLL:
            logger.debug(f"Skipping face {i}: too small ({w:.0f}x{h:.0f})")
            continue

        # Filter 2: Confidence score (optional, depends on detector)
        # if score < 0.7: # Example threshold
        #     logger.debug(f"Skipping face {i}: low confidence ({score:.2f})")
        #     continue

        # Calculate center distance (using box center)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dist_sq = (center_x - frame_center_x)**2 + (center_y - frame_center_y)**2

        candidates.append({'index': i, 'area': area, 'dist_sq': dist_sq})

    if not candidates:
        logger.debug("No suitable face candidates found after filtering.")
        return None

    # --- Selection Strategy: Prioritize largest face, use centrality as tie-breaker ---
    # Sort by area (descending), then by distance (ascending)
    candidates.sort(key=lambda x: (-x['area'], x['dist_sq']))
    best_face_idx = candidates[0]['index']

    logger.info(f"Selected face index {best_face_idx} (Area: {candidates[0]['area']:.0f}, CenterDistSq: {candidates[0]['dist_sq']:.0f})")
    return bboxes[best_face_idx], landmarks[best_face_idx]


def main():
    """Runs the face enrollment process."""
    logger.info("Initializing enrollment application...")

    try:
        detector = FaceDetector()
        embedder = FaceEmbedder()
        db = EmbeddingDatabase()
    except (FileNotFoundError, Exception) as e:
        logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
        print(f"ERROR: Failed to initialize models or database. Check logs and model paths. Details: {e}")
        return # Cannot proceed

    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        logger.critical("Could not access webcam.")
        print("ERROR: Could not access webcam. Check camera connection and permissions.")
        return

    logger.info("Webcam opened successfully.")
    print(f"\n{config.ENROLL_WINDOW_TITLE}")
    print(f"Press '{chr(config.ENROLL_CONFIRM_KEY)}' to save the selected face (green box).")
    print(f"Press '{chr(config.QUIT_KEY)}' to quit.")

    window_name = config.ENROLL_WINDOW_TITLE

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame from webcam.")
                print("ERROR: Failed to get frame from camera.")
                time.sleep(0.5) # Avoid busy-looping on error
                continue

            display_frame = frame.copy()
            frame_shape = frame.shape[:2] # (height, width)

            # --- Face Detection ---
            bboxes, landmarks = detector.detect_faces(frame)

            selected_bbox = None
            selected_landmark = None
            best_face_info = select_best_face(bboxes, landmarks, frame_shape)

            if best_face_info is not None:
                selected_bbox, selected_landmark = best_face_info

            # --- Draw Detections ---
            if bboxes is not None:
                for i, box in enumerate(bboxes):
                    x1, y1, x2, y2, _ = box.astype(int)
                    is_selected = (selected_bbox is not None) and np.array_equal(box, selected_bbox)
                    color = (0, 255, 0) if is_selected else (0, 0, 255) # Green if selected, Red otherwise
                    thickness = 2 if is_selected else 1
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # --- Display Frame ---
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF # Use mask for 64-bit systems

            # --- Handle Keystrokes ---
            if key == config.QUIT_KEY:
                logger.info("Quit key pressed. Exiting enrollment.")
                break

            elif key == config.ENROLL_CONFIRM_KEY:
                if selected_bbox is not None and selected_landmark is not None:
                    logger.info("Enrollment key pressed with a selected face.")

                    # --- Get Embedding & Quality Check ---
                    # Pass the original frame for embedding, use selected landmark
                    embedding_result = embedder.get_embedding(frame, selected_landmark, return_aligned=True)

                    if embedding_result:
                        embedding, aligned_face = embedding_result
                        if embedding is not None and aligned_face is not None:
                            # Perform quality check on the aligned face
                            if check_image_quality(aligned_face):
                                logger.info("Image quality check passed.")
                                # --- Get Person ID ---
                                try:
                                    person_id = input(">>> Enter person ID (name) to register (leave blank to cancel): ").strip()
                                    if person_id:
                                        # --- Add to Database ---
                                        db.add_embedding(person_id, embedding)
                                        # Save immediately after successful enrollment
                                        if db.save_to_file():
                                            print(f"--- Face embedding for '{person_id}' saved successfully! ---")
                                            logger.info(f"Successfully enrolled and saved '{person_id}'.")
                                        else:
                                            print(f"--- Failed to save database for '{person_id}'. Check logs. ---")
                                            logger.error(f"Failed to save database after enrolling '{person_id}'.")
                                    else:
                                        print("--- Enrollment cancelled (no ID entered). ---")
                                        logger.info("Enrollment cancelled by user (empty ID).")
                                except EOFError:
                                    print("\n--- Input interrupted. Enrollment cancelled. ---")
                                    logger.warning("EOFError during input, cancelling enrollment.")
                                    break # Exit loop if input stream closes
                            else:
                                print("--- Enrollment failed: Image quality too low (e.g., blurry). Try again. ---")
                                logger.warning("Enrollment aborted due to poor image quality.")
                        else:
                            print("--- Enrollment failed: Could not extract embedding. Try again. ---")
                            logger.error("get_embedding returned None embedding or aligned_face.")
                    else:
                        print("--- Enrollment failed: Could not extract embedding. Try again. ---")
                        logger.error("get_embedding returned None tuple.")

                else:
                    print("--- Press 's' only when a face is detected and selected (green box). ---")
                    logger.warning("Enrollment key pressed but no face was selected.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.")
        print("\nEnrollment interrupted.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during enrollment: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}. Check logs.")
    finally:
        # --- Release Resources ---
        if cap.isOpened():
            cap.release()
            logger.info("Webcam released.")
        cv2.destroyAllWindows()
        logger.info("OpenCV windows destroyed.")

if __name__ == "__main__":
    main()