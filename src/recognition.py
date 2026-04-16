# src/recognition.py
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from src import config

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Performs face recognition by comparing a query embedding against a database.
    """
    def __init__(self, threshold: float = config.RECOGNITION_THRESHOLD):
        """
        Initializes the FaceRecognizer.

        Args:
            threshold: The cosine similarity threshold for considering a match.
                       Value should be between -1.0 and 1.0.
        """
        if not -1.0 <= threshold <= 1.0:
            logger.warning(f"Recognition threshold {threshold} outside valid range [-1, 1]. Clamping.")
            self.threshold = np.clip(threshold, -1.0, 1.0)
        else:
            self.threshold = threshold
        logger.info(f"FaceRecognizer initialized with threshold: {self.threshold:.2f}")

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two 1D embedding vectors.

        Args:
            emb1: First embedding vector (1D numpy array).
            emb2: Second embedding vector (1D numpy array).

        Returns:
            The cosine similarity score (float between -1.0 and 1.0).
            Returns 0.0 if either vector has zero norm.
        """
        # Ensure inputs are numpy arrays and flatten just in case
        emb1 = np.asarray(emb1).flatten()
        emb2 = np.asarray(emb2).flatten()

        if emb1.ndim != 1 or emb2.ndim != 1:
            logger.warning("Cosine similarity requires 1D arrays.")
            return 0.0 # Cannot compute similarity

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            # Handle zero vectors to avoid division by zero
            logger.debug("Zero norm vector encountered in cosine similarity calculation.")
            return 0.0

        # Compute dot product
        dot_product = np.dot(emb1, emb2)

        # Calculate similarity
        similarity = dot_product / (norm1 * norm2)

        # Clip result to handle potential floating-point inaccuracies
        return float(np.clip(similarity, -1.0, 1.0))

    def find_best_match(self,
                        query_embedding: np.ndarray,
                        database: Dict[str, np.ndarray]
                        ) -> Optional[Tuple[str, float]]:
        """
        Finds the best match for a query embedding within the database.

        Args:
            query_embedding: The embedding vector of the face to recognize (1D numpy array).
            database: A dictionary mapping person_id (str) to their reference
                      embedding vector (1D numpy array).

        Returns:
            A tuple (person_id, similarity_score) if a match above the threshold
            is found, otherwise None.
        """
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
            logger.error("Invalid query embedding format. Must be a 1D numpy array.")
            return None
        if not isinstance(database, dict) or not database:
            # logger.debug("Recognition database is empty or invalid.")
            return None # Cannot find match in empty database

        best_match_id: Optional[str] = None
        best_similarity: float = -1.0 # Initialize below possible similarity range

        query_embedding_np = np.asarray(query_embedding) # Ensure numpy array

        for person_id, db_embedding in database.items():
            if not isinstance(db_embedding, np.ndarray) or db_embedding.ndim != 1:
                logger.warning(f"Skipping invalid database entry for '{person_id}'. Embedding is not a 1D numpy array.")
                continue

            db_embedding_np = np.asarray(db_embedding) # Ensure numpy array

            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding_np, db_embedding_np)

            # Update best match if current similarity is higher
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id

        # Return the best match only if the similarity meets the threshold
        if best_match_id is not None and best_similarity >= self.threshold:
            logger.debug(f"Best match found: '{best_match_id}' with similarity {best_similarity:.3f}")
            return best_match_id, best_similarity
        elif best_match_id is not None:
            # Log if a face was found but below threshold
            logger.debug(f"Closest match '{best_match_id}' below threshold (Sim: {best_similarity:.3f} < {self.threshold:.3f})")
            return None # Below threshold
        else:
            # No entries were compared (e.g., all db entries invalid)
            logger.debug("No suitable match found in the database.")
            return None