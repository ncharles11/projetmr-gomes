# src/config.py
import os
import logging

# --- General ---
APP_NAME = "FacialRecognitionApp"

# --- Logging ---
LOG_LEVEL = logging.WARNING  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
LOG_TO_FILE = False # Set to True to log to a file
LOG_FILENAME = "app.log"

# --- Model Paths ---
MODELS_DIR = "models"
# Make sure these models exist in the MODELS_DIR
DETECTOR_MODEL = "det_10g.onnx"
EMBEDDER_MODEL = "w600k_r50.onnx"
ANTI_SPOOFING_MODEL = "2.7_80x80_MiniFASNetV2.pth"

# Construct full paths
DETECTOR_PATH = os.path.join(MODELS_DIR, DETECTOR_MODEL)
EMBEDDER_PATH = os.path.join(MODELS_DIR, EMBEDDER_MODEL)
ANTI_SPOOFING_PATH = os.path.join(MODELS_DIR, ANTI_SPOOFING_MODEL)

# --- Database ---
DB_DIR = "data"
DB_FILENAME = "embeddings.npz"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
DB_UPDATE_ALPHA = 0.1 # Moving average factor for embedding updates

# --- Recognition ---
RECOGNITION_THRESHOLD = 0.50 # Cosine similarity threshold for a match
ADAPTATION_THRESHOLD = 0.65  # Min similarity to update embedding (continuous learning)

# --- Image Quality ---
BLUR_THRESHOLD = 75.0       # Min Laplacian variance to accept image quality for adaptation/enrollment

# --- Pose / Distance Gate ---------------------------------------------
MAX_POSE_DEG = 30        # |yaw|, |pitch|, |roll| must be ≤ this
MIN_COMPLETENESS = 0.95  # ≥ 95 % of the face bbox inside the frame
FACE_MARGIN_PX = 15      # guard band for the completeness test

YAW_OFFSET_DEG = 50.0 # Fixed bias to subtract/add to the raw yaw so that a
                      # straight-on face reads 0 °.  Calibrate once: look
                      # straight ahead, average raw yaw, set that value here.
                      # Positive if raw yaw is +N deg when looking forward,
                      # negative if it is –N deg.**

MIN_FACE_FRAC = 0.20  # face-height must be ≥ 20 % of frame height

# --- Detection ---
DETECTOR_INPUT_SIZE = (640, 640) # Must be tuple (width, height) or single int
DETECTION_INTERVAL_SECONDS = 0.3 # Process frames for detection every X seconds

# --- Anti-Spoofing ---
ANTI_SPOOFING_INPUT_SIZE = (80, 80) # Must be tuple (width, height) or single int
LIVENESS_THRESHOLD = 0.5            # Threshold for anti-spoofing detection

# --- Performance ---
# Options: ["CPUExecutionProvider"], ["CUDAExecutionProvider"], ["TensorrtExecutionProvider"] etc.
# Ensure the appropriate onnxruntime package is installed (onnxruntime or onnxruntime-gpu)
EXECUTION_PROVIDERS = ["CPUExecutionProvider"]
# Set to 0 for CPU, check GPU availability for CUDA/TensorRT
CTX_ID = 0

ANTI_SPOOFING_GPU_ID = 0 # GPU ID for anti-spoofing model, -1 for CPU

# --- GUI ---
GUI_UPDATE_INTERVAL_MS = 33 # Target ~30 FPS (milliseconds)
GUI_WINDOW_TITLE = "Facial Recognition"

# --- Serial Communication ---
SERIAL_PORT = '/dev/ttyUSB0'  # Port ESP32 sur Raspberry Pi
SERIAL_BAUDRATE = 115200

# --- Enrollment ---
ENROLL_CONFIRM_KEY = ord('s')
QUIT_KEY = ord('q')
ENROLL_WINDOW_TITLE = "Enrollment - Center Face & Press 's' to Save - 'q' to Quit"
MIN_FACE_SIZE_ENROLL = 50 # Minimum width/height in pixels for a face to be considered for enrollment

# --- Create Directories if they don't exist ---
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)