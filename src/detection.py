# src/detection.py
import os
import numpy as np
import insightface
import logging
from typing import Optional, Tuple, Union
from src import config

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Handles face detection using an InsightFace detection model.

    Attributes:
        detector: The loaded InsightFace detection model instance.
        input_size: The expected input size for the model (width, height).
    """
    def __init__(self,
                model_path: str = config.DETECTOR_PATH,
                input_size: Union[int, Tuple[int, int]] = config.DETECTOR_INPUT_SIZE,
                providers: list[str] = config.EXECUTION_PROVIDERS,
                ctx_id: int = config.CTX_ID):
        """
        Initializes the face detection model.

        Args:
            model_path: Path to the ONNX detection model file.
            input_size: Target input size (width, height) for the detector.
                        Can be an int (for square) or tuple (width, height).
            providers: List of ONNX Runtime execution providers.
            ctx_id: Context ID for the execution provider (e.g., 0 for CPU).

        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: If model loading or preparation fails.
        """
        if not os.path.exists(model_path):
            logger.error(f"Detection model file not found at: {model_path}")
            raise FileNotFoundError(f"Detection model file not found at: {model_path}")

        # Ensure input_size is a tuple (width, height)
        if isinstance(input_size, int):
            self.input_size: Tuple[int, int] = (input_size, input_size)
        elif isinstance(input_size, tuple) and len(input_size) == 2:
            self.input_size = input_size
        else:
            logger.error(f"Invalid input_size format: {input_size}. Using default (640, 640).")
            self.input_size = (640, 640) # Fallback default

        try:
            # Using model_zoo is convenient for standard models
            self.detector = insightface.model_zoo.get_model(model_path, providers=providers)
            # Set detection threshold if needed, e.g., self.detector.det_thresh = 0.5
            # Some models might require specific setup parameters
            self.detector.prepare(ctx_id=ctx_id, input_size=self.input_size) # Pass input size here
            logger.info(f"Face detector initialized successfully with model: {model_path} using {providers}")
        except Exception as e:
            logger.error(f"Failed to initialize detector model '{model_path}': {e}", exc_info=True)
            # Re-raise the exception to signal failure
            raise

    def detect_faces(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detects faces in the given frame.

        Args:
            frame: The input image frame (BGR color format expected).

        Returns:
            A tuple containing:
            - bboxes (np.ndarray): Array of bounding boxes (x1, y1, x2, y2, score)
                                   or None if no faces detected or error occurred.
            - landmarks (np.ndarray): Array of landmarks (5 points per face, shape [N, 5, 2])
                                      or None if no faces detected or error occurred.
                                      Order matches bboxes.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            logger.warning("Invalid frame provided for detection (must be 3D numpy array).")
            return None, None

        try:
            # The detect method takes the frame and optional parameters
            # Some detector models might have parameters like `threshold` or `nms_thresh`
            # Check the specific model's documentation or the insightface source if needed.
            bboxes, landmarks = self.detector.detect(frame,
                                                    # input_size=self.input_size, # Already set in prepare
                                                    max_num=0, # 0 means no limit
                                                    metric='default') # 'default' or 'max'

            if bboxes is None or bboxes.shape[0] == 0:
                return None, None # Return None explicitly if no faces found

            # Ensure landmarks are also returned if bboxes are found
            if landmarks is None or landmarks.shape[0] != bboxes.shape[0]:
                logger.warning(f"Mismatch between bbox ({bboxes.shape[0]}) and landmark ({landmarks.shape[0] if landmarks is not None else 'None'}) counts.")
                return None, None # Return None for both if mismatch

            logger.debug(f"Detected {bboxes.shape[0]} faces.")
            return bboxes, landmarks

        except Exception as e:
            logger.error(f"Error during face detection: {e}", exc_info=True)
            return None, None # Return None for both on error