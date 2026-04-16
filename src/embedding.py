# src/embedding.py
import os
import numpy as np
import insightface
import cv2
import logging
from insightface.utils import face_align
from typing import Optional, Tuple
from src import config

logger = logging.getLogger(__name__)

class FaceEmbedder:
    """
    Handles face alignment and embedding generation using an InsightFace model.

    Attributes:
        embedder: The loaded InsightFace embedding model instance.
    """
    def __init__(self,
                 model_path: str = config.EMBEDDER_PATH,
                 providers: list[str] = config.EXECUTION_PROVIDERS,
                 ctx_id: int = config.CTX_ID):
        """
        Initializes the face embedding model.

        Args:
            model_path: Path to the ONNX embedding model file.
            providers: List of ONNX Runtime execution providers.
            ctx_id: Context ID for the execution provider (e.g., 0 for CPU).

        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: If model loading or preparation fails.
        """
        if not os.path.exists(model_path):
            logger.error(f"Embedding model file not found at: {model_path}")
            raise FileNotFoundError(f"Embedding model file not found at: {model_path}")

        try:
            self.embedder = insightface.model_zoo.get_model(model_path, providers=providers)
            self.embedder.prepare(ctx_id=ctx_id)
            logger.info(f"Face embedder initialized successfully with model: {model_path} using {providers}")
        except Exception as e:
            logger.error(f"Failed to initialize embedder model '{model_path}': {e}", exc_info=True)
            # Re-raise the exception to signal failure to the caller
            raise

    def get_embedding(self,
                      frame: np.ndarray,
                      landmark: np.ndarray,
                      return_aligned: bool = False
                      ) -> Optional[np.ndarray | Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Computes the embedding for a face defined by landmarks in a frame.

        Performs alignment using insightface.utils.face_align.norm_crop.
        Normalizes the resulting embedding to unit length.

        Args:
            frame: The input image frame (BGR color format expected).
            landmark: A (5, 2) numpy array of facial landmarks (x, y).
            return_aligned: If True, also return the 112x112 aligned face image.

        Returns:
            If return_aligned is False:
                - A 1D numpy vector (normalized embedding) if successful.
                - None otherwise.
            If return_aligned is True:
                - A tuple (normalized_embedding, aligned_face) if successful.
                - A tuple (None, None) otherwise.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            logger.warning("Invalid frame provided for embedding (must be 3D numpy array).")
            return (None, None) if return_aligned else None
        if not isinstance(landmark, np.ndarray) or landmark.shape != (5, 2):
            logger.warning(f"Invalid landmark data provided (shape: {landmark.shape if landmark is not None else 'None'}). Expected (5, 2).")
            return (None, None) if return_aligned else None

        try:
            # Align the face using the provided landmarks.
            # norm_crop creates a 112x112 aligned face image (RGB)
            # Important: norm_crop expects BGR input frame, landmarks as float/int
            aligned_face = face_align.norm_crop(frame, landmark=landmark.astype(np.int32))

            if aligned_face is None or aligned_face.size == 0:
                logger.warning("Face alignment failed (norm_crop returned None or empty).")
                return (None, None) if return_aligned else None

            # Get embedding from the aligned face chip
            # get_feat expects the aligned image (RGB, shape (112, 112, 3))
            # Models might expect input in range [0,1] or [-1,1], but insightface handles this often.
            embedding = self.embedder.get_feat(aligned_face)

            if embedding is None:
                logger.warning("Embedder model returned None for the aligned face.")
                return (None, None) if return_aligned else None

            # Ensure the output is a flattened 1D array and normalize it
            embedding_flat = embedding.flatten()
            norm = np.linalg.norm(embedding_flat)
            normalized_embedding = embedding_flat / norm if norm > 1e-6 else embedding_flat # Avoid division by zero

            if return_aligned:
                return normalized_embedding, aligned_face
            else:
                return normalized_embedding

        except cv2.error as cv_err:
            logger.error(f"OpenCV error during face alignment: {cv_err}", exc_info=True)
            return (None, None) if return_aligned else None
        except Exception as e:
            # Catch potential errors from get_feat or other unexpected issues
            logger.error(f"Error computing embedding: {str(e)}", exc_info=True)
            return (None, None) if return_aligned else None