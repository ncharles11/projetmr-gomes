# src/main.py
import tkinter as tk
import cv2
import time
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

# --- Setup Logging First ---
try:
    from src.log_config import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    logging.warning(".log_config not found, using basic logging.")

# --- Import project modules AFTER logging is set up ---
from src.detection import FaceDetector
from src.antispoofing import antispoofing
from src.embedding import FaceEmbedder
from src.recognition import FaceRecognizer
from src.database import EmbeddingDatabase
from src.gui import FaceRecognitionGUI
from src import config # Import configuration
from src.utils import check_image_quality, draw_detection, estimate_pose

logger = logging.getLogger(__name__)

# --- Main Application Class ---
class RecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.gui: Optional[FaceRecognitionGUI] = None
        self.detector: Optional[FaceDetector] = None
        self.antispoofing: Optional[antispoofing.AntiSpoofingPredictor] = None
        self.embedder: Optional[FaceEmbedder] = None
        self.recognizer: Optional[FaceRecognizer] = None
        self.db: Optional[EmbeddingDatabase] = None
        self.cap: Optional[cv2.VideoCapture] = None

        # Application state
        self.state: Dict[str, Any] = {
            "last_detection_time": 0.0,
            "last_save_time": time.time(),
            "current_status": "Initializing...",
            "running": True,
            "last_processed_frame": None # Store the frame with drawings
        }

        if not self._initialize_components():
            self.state["running"] = False
            # GUI might not be fully initialized, so print error too
            print("ERROR: Application initialization failed. Check logs.")
            # Attempt to show error in GUI if it exists
            if self.gui:
                self.gui.update_status("ERROR: Initialization Failed!")
            else:
                # If GUI failed, need to handle root window closure
                root.destroy()
            return

        # Setup GUI and Video Capture
        self.gui = FaceRecognitionGUI(root)
        self._setup_video_capture()

        # Set close protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _initialize_components(self) -> bool:
        """Initializes face detection, anti-spoofing, embedding, and recognition components."""
        logger.info("Initializing application components...")
        try:
            self.detector = FaceDetector()
            self.antispoofing = antispoofing.AntiSpoofingPredictor()
            self.embedder = FaceEmbedder()
            self.recognizer = FaceRecognizer()
            self.db = EmbeddingDatabase()
            logger.info("All components initialized successfully.")
            return True
        except (FileNotFoundError, Exception) as e:
            logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
            return False

    def _setup_video_capture(self):
        """Initializes the video capture device."""
        logger.info("Initializing video capture...")
        try:
            self.cap = cv2.VideoCapture(0) # Use 0 for default camera
            if not self.cap or not self.cap.isOpened():
                raise IOError("Could not open video device.")
            logger.info("Video capture initialized successfully.")
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera properties: {int(width)}x{int(height)} @ {fps:.2f} FPS")
        except (IOError, Exception) as e:
            logger.critical(f"Failed to initialize video capture: {e}", exc_info=True)
            self.gui.update_status(f"Error: Cannot open camera!\n{e}")
            self.state["running"] = False # Stop the main loop

    def _process_single_face(self,
                             frame: np.ndarray,
                             box: np.ndarray,
                             landmark: np.ndarray,
                             database_content: Dict[str, np.ndarray]
                             ) -> Dict[str, Any]:
        """
        Processes a single detected face: anti-spoofing, embedding, recognition, adaptation.

        Args:
            frame: The full original video frame.
            box: The bounding box [x1, y1, x2, y2, score] for this face.
            landmark: The 5 facial landmarks for this face.
            database_content: The current embeddings database.

        Returns:
            A dictionary containing processing results for this face:
            {
                "status": str,         # Status description (e.g., "Authorized", "Spoof", "Error")
                "label": str,          # Label text for the bounding box
                "color": tuple,        # BGR color for the bounding box
                "embedding": np.ndarray|None, # Calculated embedding (if live and successful)
                "aligned_face": np.ndarray|None, # Aligned face (if live and successful)
                "person_id": str|None, # Identified person ID
                "similarity": float    # Best match similarity score
            }
        """
        results = { 
            "status": "Error: Processing Failed",
            "label": "Error",
            "color": (0, 0, 255), # Red default for error
            "embedding": None,
            "aligned_face": None,
            "person_id": None,
            "similarity": -1.0
        }

        # --- 1. Face Pose, Size and Completeness Check ---
        is_ok, info = estimate_pose(frame, box[:4], landmark)

        if not is_ok:
            if not info['face_complete']:
                results["status"] = "Face Not Complete"
                results["label"]  = f"Incomplete ({info['completeness']:.1%})"
                results["color"]  = (0,165,255)   # orange

            elif not info['size_valid']:
                results["status"] = "Face Too Small"
                results["label"]  = f"Too Far ({info['face_frac']*100:.0f}%)"
                results["color"]  = (0,165,255)

            elif not info['pose_valid']:
                results["status"] = "Face Pose Invalid"
                results["label"]  = (f"Bad Pose "
                                    f"(Y:{info['yaw']:.0f}deg "
                                    f"P:{info['pitch']:.0f}deg "
                                    f"R:{info['roll']:.0f}deg)")
                results["color"]  = (0,165,255)

            else:
                results["status"] = "Face Quality Poor"
                results["label"]  = "Poor Quality"
                results["color"]  = (0,165,255)

            logger.debug(
                "Reject – Y:%+.1f deg P:%+.1f deg R:%+.1f deg "
                "complete:%5.1f %% size:%5.1f %% "
                "(inside:%s pose:%s size:%s)",
                info['yaw'], info['pitch'], info['roll'],
                info['completeness']*100, info['face_frac']*100,
                info['face_complete'], info['pose_valid'], info['size_valid']
            )
            return results
        
        # --- 2. Basic Quality Check ---
        x1, y1, x2, y2 = box[:4].astype(int)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0 and not check_image_quality(face_crop):
            results["status"] = "Face Quality Poor"
            results["label"] = "Poor Quality"
            results["color"] = (0, 165, 255)
            logger.debug("Face rejected due to poor quality (early check)")
            return results
            
        # --- 3. Anti-Spoofing Check ---
        is_live = False
        liveness_score = -1.0
        prediction_result = self.antispoofing.predict(frame, box[:4])
        if prediction_result:
            is_live, liveness_score = prediction_result
            logger.debug(f"Liveness check: Live={is_live}, Score={liveness_score:.3f}")
        else:
            logger.warning("Anti-spoofing prediction failed.")
            is_live = False # Treat failure as spoof for safety
            results["status"] = "Error: Anti-Spoof Failed"
            results["label"] = f"Spoof? ({liveness_score:.2f})" # Indicate uncertainty
            results["color"] = (0, 0, 255)

        if not is_live:
            results["status"] = "Spoof Detected"
            results["label"] = f"Spoof ({liveness_score:.2f})"
            results["color"] = (0, 0, 255) # Red
            logger.warning(f"SPOOF DETECTED! Liveness score: {liveness_score:.2f}")
            return results # Stop processing if spoof

        # --- 4. Embedding (Only if Live) ---
        results["label"] = f"Live ({liveness_score:.2f}), Embedding..."
        results["color"] = (255, 255, 0) # Cyan during embedding

        embedding_result = self.embedder.get_embedding(frame, landmark, return_aligned=True)
        if not embedding_result or embedding_result[0] is None:
            logger.warning("Embedding extraction failed (Result None or Embedding None).")
            results["status"] = "Error: Embedding Failed"
            results["label"] = f"Emb. Err ({liveness_score:.2f})"
            results["color"] = (0, 165, 255) # Orange
            return results

        embedding, aligned_face = embedding_result
        results["embedding"] = embedding
        results["aligned_face"] = aligned_face # Store for potential adaptation

        # --- 5. Recognition (Only if Embedding Successful) ---
        results["label"] = f"Live ({liveness_score:.2f}), Recognizing..."
        results["color"] = (255, 165, 0) # Blue during recognition

        match_result = self.recognizer.find_best_match(embedding, database_content)

        if match_result:
            person_id, sim = match_result
            results["status"] = f"Authorized: {person_id}"
            results["label"] = f"{person_id} ({sim:.2f})"
            results["color"] = (0, 255, 0) # Green
            results["person_id"] = person_id
            results["similarity"] = sim
            logger.info(f"Match found: {person_id} (Sim: {sim:.2f}), Live: {liveness_score:.2f}")
        else:
            results["status"] = "Unknown (Live Face)"
            results["label"] = f"Unknown ({liveness_score:.2f})"
            results["color"] = (0, 165, 255) # Orange
            results["person_id"] = None # Ensure it's None
            results["similarity"] = -1.0
            logger.info(f"Unknown live face detected. Live: {liveness_score:.2f}")

        return results

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """Detects faces and processes each one using a helper method."""
        start_time = time.time()
        processed_frame = frame.copy()
        overall_status = "No face detected" # Default status for the frame
        active_processing = False
        highest_priority_status = 10 # Lower number = higher priority

        STATUS_PRIORITY = { # Define priority for overall status message
            "Spoof Detected": 0,                           # Highest priority - security threat
            lambda s: s.startswith("Authorized"): 1,       # Successful authentication
            "Unknown (Live Face)": 2,                      # Live but unrecognized
            "Face Pose Invalid": 3,                        # Face quality issues
            "Face Not Complete": 4,                        # Face positioning issues
            "Face Too Small": 5,                           # Distance/size issues
            "Face Quality Poor": 6,                        # General quality issues
            "Error": 7,                                    # Technical errors
            "Processing...": 8,                            # Active processing
            "No face detected": 9                          # Default state
        }

        def get_status_priority(status_str):
            for key, priority in STATUS_PRIORITY.items():
                if callable(key) and key(status_str):
                    return priority
                elif isinstance(key, str) and status_str.startswith(key):
                    return priority
            return 99 # Default low priority

        # Ensure essential components are ready
        if not self.detector or not self.embedder or not self.recognizer or not self.db:
             logger.error("Core components not initialized in _process_frame.")
             return processed_frame, "Error: Components not ready"

        database_content = self.db.get_database()

        # --- 1. Detection ---
        bboxes, landmarks = self.detector.detect_faces(frame)

        if bboxes is not None and landmarks is not None:
            num_faces = bboxes.shape[0]
            logger.debug(f"Detected {num_faces} faces.")

            if num_faces > 0:
                active_processing = True
                overall_status = "Processing..." # Initial status if faces are found

                # --- 2. Process Each Detected Face ---
                for i in range(num_faces):
                    box = bboxes[i]
                    landmark = landmarks[i]

                    # Call the helper function to process this face
                    face_results = self._process_single_face(frame, box, landmark, database_content)

                    # Update overall frame status based on priority
                    current_priority = get_status_priority(face_results["status"])
                    if current_priority < highest_priority_status:
                        highest_priority_status = current_priority
                        overall_status = face_results["status"] # Set overall status to highest priority found

                    # --- 3. Adaptation Logic (Based on helper results) ---
                    if (face_results["person_id"] is not None and
                        face_results["similarity"] >= config.ADAPTATION_THRESHOLD and
                        face_results["embedding"] is not None and
                        face_results["aligned_face"] is not None):
                        logger.debug(f"Adapting embedding for {face_results['person_id']} (Sim: {face_results['similarity']:.2f})")
                        self.db.add_embedding(
                            face_results["person_id"],
                            face_results["embedding"], # Use the embedding calculated by helper
                            alpha=config.DB_UPDATE_ALPHA
                        )

                    # --- 4. Draw Detection ---
                    draw_detection(processed_frame, box[:4], landmark,
                                   face_results["label"], face_results["color"])
                    
        else: # No faces detected
            overall_status = "No face detected"
            active_processing = False

        # Log processing time only if faces were detected and processed
        if active_processing:
            processing_time = (time.time() - start_time) * 1000 # ms
            logger.debug(f"Frame processing time: {processing_time:.2f} ms")

        # Return the annotated frame and the determined overall status
        return processed_frame, overall_status


    def _update_gui_loop(self):
        """The main loop that reads frames and updates the GUI."""
        if not self.state["running"]:
            logger.info("Update loop stopped.")
            return

        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not available in update loop.")
            self.gui.update_status("Error: Camera disconnected!")
            # Schedule next attempt slightly further out
            self.root.after(1000, self._update_gui_loop)
            return

        start_read_time = time.time()
        ret, frame = self.cap.read()
        read_time = (time.time() - start_read_time) * 1000
        logger.debug(f"Frame read time: {read_time:.2f} ms")

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera.")
            # Keep trying, maybe the camera will come back
            self.root.after(config.GUI_UPDATE_INTERVAL_MS, self._update_gui_loop)
            return

        current_time = time.time()
        frame_to_display = frame.copy() # Default to raw frame

        # --- Periodic Detection/Recognition ---
        if current_time - self.state["last_detection_time"] >= config.DETECTION_INTERVAL_SECONDS:
            processed_frame, status = self._process_frame(frame)
            self.state["current_status"] = status
            self.state["last_detection_time"] = current_time
            self.state["last_processed_frame"] = processed_frame # Store frame with drawings
            frame_to_display = processed_frame # Display the processed frame
            self.gui.update_status(self.state["current_status"])
            logger.debug(f"Detection cycle finished. Status: {status}")
        else:
            # --- Display previous results between detection cycles ---
            if self.state["last_processed_frame"] is not None:
                # Showing last processed frame
                frame_to_display = self.state["last_processed_frame"]
                # Keep updating status label even between detections
                self.gui.update_status(self.state["current_status"])
            else:
                # Before first detection or if detection failed
                self.gui.update_status(self.state["current_status"])


        # --- Update GUI Image ---
        if self.gui:
            self.gui.update_image(frame_to_display)

        # --- Periodic Database Save ---
        # Save every 5 minutes if dirty
        if self.db and current_time - self.state["last_save_time"] > 300:
            logger.debug("Periodic save check...")
            self.db.save_if_dirty() # Saves only if needed
            self.state["last_save_time"] = current_time # Update time even if not saved

        # --- Schedule next update ---
        self.root.after(config.GUI_UPDATE_INTERVAL_MS, self._update_gui_loop)


    def run(self):
        """Starts the GUI main loop and the update process."""
        if not self.state["running"]:
            logger.warning("Application not started due to initialization errors.")
            # Ensure window closes if it was created but app failed
            if self.root and not self.gui:
                self.root.destroy()
            return

        logger.info("Starting GUI main loop...")
        self.state["current_status"] = "Waiting for detection..."
        self.gui.update_status(self.state["current_status"])
        # Start the update loop
        self.root.after(50, self._update_gui_loop) # Start after small delay
        # Start Tkinter main loop
        self.root.mainloop()

    def _on_close(self):
        """Handles cleanup when the GUI window is closed."""
        logger.info("Close requested. Shutting down...")
        self.state["running"] = False # Signal update loop to stop

        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Video capture released.")

        # Save database if needed
        if self.db:
            logger.info("Saving database if changes were made...")
            self.db.save_if_dirty()

        # Destroy the Tkinter window
        if self.root:
            self.root.destroy()
            logger.info("GUI destroyed.")

        logger.info("Application shutdown complete.")

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    app.run()