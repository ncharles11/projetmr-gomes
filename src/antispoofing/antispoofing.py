import os
import numpy as np
import logging
from typing import Optional, Tuple, List

from src import config
from .anti_spoof_predict import AntiSpoofPredict
from .generate_patches import CropImage
from .utils import parse_model_name

logger = logging.getLogger(__name__)

class AntiSpoofingPredictor:
    """
    Performs anti-spoofing (liveness detection) using a specified PyTorch model.
    Handles cropping, preprocessing, and inference.
    """
    def __init__(self,
                 model_path: str = config.ANTI_SPOOFING_PATH,
                 device_id: int = config.ANTI_SPOOFING_GPU_ID):
        """
        Initializes the AntiSpoofingPredictor.

        Args:
            model_path (str): Path to the PyTorch anti-spoofing model (.pth).
            device_id (int): GPU device ID (-1 for CPU).

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the model name cannot be parsed.
            Exception: For other initialization errors.
        """
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            logger.error(f"Anti-spoofing model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Anti-spoofing model file not found at: {self.model_path}")

        logger.info(f"Initializing AntiSpoofingPredictor with PyTorch model: {self.model_path}")

        # --- Parse Model Parameters ---
        try:
            self.model_name = os.path.basename(self.model_path)
            h, w, _, scale = parse_model_name(self.model_name)
            self.target_height: int = h
            self.target_width: int = w
            # Ensure scale is float or None
            self.scale: Optional[float] = float(scale) if scale is not None else None
            self.crop_enabled: bool = self.scale is not None

            logger.info(f"Parsed parameters: Size=({self.target_width}x{self.target_height}), Crop Scale={self.scale}")
        except Exception as e:
            logger.error(f"Failed to parse model name '{self.model_name}': {e}", exc_info=True)
            raise ValueError(f"Could not parse parameters from model name: {self.model_name}") from e

        # --- Initialize Helpers ---
        try:
            # AntiSpoofPredict handles device selection
            self.model_predictor = AntiSpoofPredict(device_id)
            self.image_cropper = CropImage()
            logger.info(f"Underlying AntiSpoofPredict initialized for device_id={device_id}.")
        except Exception as e:
            logger.error(f"Failed to initialize AntiSpoofPredict/CropImage: {e}", exc_info=True)
            raise
    
    def predict(self,
                original_frame: np.ndarray,
                face_bbox: np.ndarray,
               ) -> Optional[Tuple[bool, float]]:
        """
        Performs cropping (if applicable) and predicts if the face is live or spoof.

        Args:
            original_frame: The full original BGR video frame.
            face_bbox: The bounding box [x1, y1, x2, y2] as a numpy array.

        Returns:
            Tuple (is_live, liveness_score) or None if processing/prediction fails.
            is_live (bool): True if predicted live, False if predicted spoof.
            liveness_score (float): The liveness score associated with the prediction.
        """
        # Basic input validation
        if not isinstance(original_frame, np.ndarray) or original_frame.ndim != 3:
            logger.warning("Invalid original_frame provided for anti-spoofing.")
            return None
        if not (isinstance(face_bbox, np.ndarray)) or len(face_bbox) != 4:
            logger.warning(f"Invalid face_bbox provided (expected array of 4 values): {face_bbox}")
            return None
        
        try:
            cropped_image = self.image_cropper.crop(
                org_img=original_frame,
                bbox=face_bbox,
                scale=self.scale,
                out_w=self.target_width,
                out_h=self.target_height,
                crop=self.crop_enabled
            )

            if cropped_image is None:
                logger.debug("Image cropping resulted in None.")
                return None
            
            # --- 2. Run Prediction ---
            # The underlying AntiSpoofPredict handles preprocessing and inference
            prediction_output = self.model_predictor.predict(cropped_image, self.model_path)

            if prediction_output is None:
                logger.warning("Underlying model prediction returned None.")
                return None
            
            # --- 3. Interpret Results ---
            scores = prediction_output.flatten()
            if len(scores) < 2:
                logger.warning(f"Prediction output has insufficient scores: {scores}")
                return None

            predicted_label = np.argmax(scores)
            live_class_index = 1 # 1 = live
            is_live = (predicted_label == live_class_index)

            # Use the score corresponding to the predicted label, scaled as per original code
            liveness_score = float(scores[predicted_label])

            logger.debug(f"Anti-spoofing raw scores: {scores}, PredLabel: {predicted_label}, IsLive: {is_live}, FinalScore: {liveness_score:.4f}")

            return is_live, liveness_score

        except Exception as e:
            # Log general errors during the process
            logger.error(f"Error during anti-spoofing prediction for bbox {face_bbox}: {e}", exc_info=True)
            return None
