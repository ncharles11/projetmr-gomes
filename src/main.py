import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tkinter as tk
from tkinter import simpledialog
import cv2
import time
import numpy as np
import logging
import json
import os
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
from src.hardware_comm import ESP32Communicator

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

        # --- User Profiles & Anti-Spam ---
        self.dernier_utilisateur_actif = None
        self.utilisateurs_coords = {}
        self._load_utilisateurs_json()
        self.esp32_comm = ESP32Communicator()

        # Application state
        self.is_enrolling = False
        self.is_authenticated = False
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
        self.gui.on_motor_manual = self._on_motor_manual
        self.gui.on_calibrate_zero = self._on_calibrate_zero
        self._setup_video_capture()

        # Set close protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_utilisateurs_json(self):
        """Loads user coordinates from data/utilisateurs.json."""
        json_path = os.path.join("data", "utilisateurs.json")
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.utilisateurs_coords = json.load(f)
                logger.info(f"Loaded coordinates for {len(self.utilisateurs_coords)} users.")
            else:
                logger.warning(f"File {json_path} not found. Using empty profiles.")
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")

    def _save_utilisateurs_json(self):
        """Saves user coordinates to data/utilisateurs.json."""
        json_path = os.path.join("data", "utilisateurs.json")
        try:
            # S'assurer que le dossier data existe
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(self.utilisateurs_coords, f, indent=4)
            logger.info(f"Saved {len(self.utilisateurs_coords)} user profiles to {json_path}.")
            return True
        except Exception as e:
            logger.error(f"Error saving {json_path}: {e}")
            return False

    def _enroll_new_user(self, temps_droite, temps_bas):
        """Procédure d'enrôlement pour un nouvel utilisateur."""
        self.is_enrolling = True
        self.gui.update_status("ENRÔLEMENT EN COURS...")
        
        # 1. Demander le nom
        person_id = simpledialog.askstring("Nouveau Conducteur", "Entrez le nom du nouveau conducteur :", parent=self.root)
        
        if not person_id:
            logger.info("Enrôlement annulé (pas de nom fourni).")
            self.is_enrolling = False
            return

        person_id = person_id.strip()
        
        # 2. Capturer une image de qualité
        logger.info(f"Démarrage de la capture pour {person_id}...")
        start_enroll_time = time.time()
        timeout = 10 # 10 secondes pour trouver un visage
        
        captured_embedding = None
        
        while time.time() - start_enroll_time < timeout:
            ret, frame = self.cap.read()
            if not ret: continue
            
            bboxes, landmarks = self.detector.detect_faces(frame)
            if bboxes is not None and len(bboxes) > 0:
                # Prendre le premier visage pour simplifier
                landmark = landmarks[0]
                embedding_result = self.embedder.get_embedding(frame, landmark, return_aligned=True)
                
                if embedding_result and embedding_result[0] is not None:
                    embedding, aligned_face = embedding_result
                    if check_image_quality(aligned_face):
                        captured_embedding = embedding
                        logger.info(f"Visage capturé avec succès pour {person_id}.")
                        break
            
            # Afficher le flux pendant la capture
            self.gui.update_image(frame)
            self.root.update()
            time.sleep(0.01)

        if captured_embedding is not None:
            # 3. Sauvegarder l'embedding
            self.db.add_embedding(person_id, captured_embedding)
            self.db.save_to_file()
            
            # 4. Sauvegarder les réglages moteurs
            self.utilisateurs_coords[person_id] = {
                "tempsDroite": temps_droite,
                "tempsBas": temps_bas
            }
            self._save_utilisateurs_json()
            
            self.dernier_utilisateur_actif = person_id
            self.gui.update_status(f"Bienvenue {person_id} !")
            logger.info(f"Profil complet créé pour {person_id}.")
        else:
            self.gui.update_status("Erreur : Aucun visage détecté.")
            logger.error("Échec de l'enrôlement : timeout ou mauvaise qualité.")

        self.is_enrolling = False

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

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[str]]:
        """Detects faces and processes each one using a helper method."""
        start_time = time.time()
        processed_frame = frame.copy()
        overall_status = "No face detected" # Default status for the frame
        active_processing = False
        highest_priority_status = 10 # Lower number = higher priority
        active_person_id = None

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
             return processed_frame, "Error: Components not ready", None

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
                        active_person_id = face_results["person_id"]

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

        # Return the annotated frame, the determined overall status, and identified person_id
        return processed_frame, overall_status, active_person_id

    def _handle_esp32_messages(self):
        """Vérifie et gère les messages reçus de l'ESP32 de manière non-bloquante."""
        if not hasattr(self, 'esp32_comm') or self.esp32_comm is None:
            return

        try:
            msg = self.esp32_comm.read_message()
            if not msg:
                return

            action = msg.get("action")
            if action == "save_profile":
                temps_droite = msg.get("tempsDroite", 0)
                temps_bas = msg.get("tempsBas", 0)
                
                if self.dernier_utilisateur_actif is None:
                    logger.info("Commande de sauvegarde reçue pour un utilisateur inconnu.")
                    self._enroll_new_user(temps_droite, temps_bas)
                else:
                    user = self.dernier_utilisateur_actif
                    logger.info(f"Mise à jour des réglages pour {user}: R={temps_droite}, D={temps_bas}")
                    self.utilisateurs_coords[user] = {
                        "tempsDroite": temps_droite,
                        "tempsBas": temps_bas
                    }
                    self._save_utilisateurs_json()
                    self.gui.update_status(f"Réglages mis à jour pour {user}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des messages ESP32: {e}")

    def _on_motor_manual(self, direction: str):
        """Envoie le caractère de commande moteur correspondant à la direction du pavé."""
        char_map = {"up": "H", "down": "B", "left": "G", "right": "D"}
        char = char_map.get(direction)
        if char is None:
            logger.warning(f"Direction inconnue reçue du pavé : {direction!r}")
            return
        logger.info(f"Commande manuelle moteur : {direction} → '{char}'")
        self.esp32_comm.send_raw(char)

    def _on_calibrate_zero(self):
        """Envoie le caractère 'Z' à l'ESP32 pour fixer la position zéro des moteurs."""
        logger.info("Calibration : envoi du point zéro 'Z' à l'ESP32.")
        self.esp32_comm.send_raw("Z")

    def _update_gui_loop(self):
        """The main loop that reads frames and updates the GUI with intensive logging."""
        try:
            logger.info("--- DEBUG LOOP START ---")
            
            if not self.state["running"]:
                logger.info("Update loop stopped.")
                return

            # Gérer les messages ESP32 (ex: demande de sauvegarde)
            self._handle_esp32_messages()
            logger.info("ESP32 messages handled.")

            if self.is_enrolling:
                logger.info("Skipping recognition (enrolling)...")
                self.root.after(config.GUI_UPDATE_INTERVAL_MS, self._update_gui_loop)
                return

            is_open = self.cap.isOpened() if self.cap else False
            logger.info(f"Camera open state: {is_open}")

            if not self.cap or not is_open:
                logger.error("Camera not available in update loop.")
                self.gui.update_status("Error: Camera disconnected!")
                self.root.after(1000, self._update_gui_loop)
                return

            start_read_time = time.time()
            ret, frame = self.cap.read()
            read_time = (time.time() - start_read_time) * 1000
            
            logger.info(f"Camera read: ret={ret}, frame_is_none={frame is None}")

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera. Retrying...")
                self.root.after(33, self._update_gui_loop)
                return
            
            logger.info(f"Frame shape: {frame.shape}")

            current_time = time.time()
            frame_to_display = frame.copy() # Default to raw frame

            # --- Periodic Detection/Recognition (désactivée après authentification) ---
            if not self.is_authenticated:
                if current_time - self.state["last_detection_time"] >= config.DETECTION_INTERVAL_SECONDS:
                    logger.info("Starting detection cycle...")
                    processed_frame, status, person_id = self._process_frame(frame)
                    self.state["current_status"] = status
                    self.state["last_detection_time"] = current_time
                    self.state["last_processed_frame"] = processed_frame
                    frame_to_display = processed_frame
                    logger.info(f"Detection cycle finished. Status: {status}")

                    if person_id is not None:
                        # --- Authentification réussie — une seule fois ---
                        self.is_authenticated = True
                        self.dernier_utilisateur_actif = person_id

                        # Recharger le JSON pour avoir les réglages les plus récents
                        self._load_utilisateurs_json()
                        if person_id in self.utilisateurs_coords:
                            user_data = self.utilisateurs_coords[person_id]
                            temps_droite = user_data.get("tempsDroite", 0)
                            temps_bas = user_data.get("tempsBas", 0)
                            logger.info(
                                f"Réglages chargés pour '{person_id}' : "
                                f"tempsDroite={temps_droite}, tempsBas={temps_bas}"
                            )
                        else:
                            temps_droite = 0
                            temps_bas = 0
                            logger.warning(
                                f"'{person_id}' absent de utilisateurs.json. "
                                f"Envoi des valeurs par défaut (tempsDroite=0, tempsBas=0)."
                            )

                        welcome_msg = f"Bonjour {person_id} !"
                        self.state["current_status"] = welcome_msg
                        self.gui.update_status(welcome_msg)
                        logger.info(f"Authentification réussie : {person_id}. Envoi réglages ESP32.")
                        self.esp32_comm.send_auth_data(person_id, temps_droite, temps_bas)
                    else:
                        self.gui.update_status(self.state["current_status"])
                else:
                    # Entre deux cycles — affiche la dernière frame annotée
                    if self.state["last_processed_frame"] is not None:
                        frame_to_display = self.state["last_processed_frame"]
                    self.gui.update_status(self.state["current_status"])
            else:
                # Authentifié — pas de détection, flux vidéo brut, statut figé
                frame_to_display = frame
                self.gui.update_status(self.state["current_status"])

            # --- Update GUI Image ---
            if self.gui:
                logger.info("Calling gui.update_image...")
                self.gui.update_image(frame_to_display)
                logger.info("gui.update_image finished.")

            # --- Periodic Database Save ---
            if self.db and current_time - self.state["last_save_time"] > 300:
                logger.debug("Periodic save check...")
                self.db.save_if_dirty()
                self.state["last_save_time"] = current_time

            # --- Schedule next update ---
            self.root.after(config.GUI_UPDATE_INTERVAL_MS, self._update_gui_loop)
            logger.info("--- DEBUG LOOP END (Scheduled) ---")

        except Exception as e:
            logger.error(f"FATAL ERROR in _update_gui_loop: {e}", exc_info=True)
            # Re-planifier quand même pour éviter que l'app ne s'arrête
            self.root.after(1000, self._update_gui_loop)

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
        
        # Forcer le rendu initial sur macOS
        self.root.update_idletasks()
        self.root.update()

        # Start the update loop
        self.root.after(1500, self._update_gui_loop) # Délai macOS : laisse les drivers caméra s'initialiser
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
