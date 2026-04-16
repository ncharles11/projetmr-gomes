# src/database.py
import os
import numpy as np
import logging
from typing import Dict, Optional
from src import config

logger = logging.getLogger(__name__) # Get logger for this module

class EmbeddingDatabase:
    """
    Manages storing and retrieving face embeddings using a compressed NumPy file.

    Attributes:
        db_path (str): Path to the database file.
        db (Dict[str, np.ndarray]): In-memory dictionary holding person_id -> embedding.
        dirty (bool): Flag indicating if the database has unsaved changes.
    """
    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the EmbeddingDatabase.

        Args:
            db_path: Path to the database file. Defaults to config.DB_PATH.
        """
        self.db_path = db_path or config.DB_PATH
        self.db: Dict[str, np.ndarray] = {}
        self.dirty = False

        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir: # Check if directory part exists (not just filename)
            os.makedirs(db_dir, exist_ok=True)

        # Load the database if it exists
        self.load_from_file()

    def add_embedding(self, person_id: str, embedding: np.ndarray, alpha: float = config.DB_UPDATE_ALPHA):
        """
        Adds or updates an embedding for a person_id.

        If the person_id exists, applies exponential moving average:
            new_emb = (1 - alpha) * old_emb + alpha * new_emb
        Normalizes the resulting embedding to unit length.

        Args:
            person_id: The identifier for the person.
            embedding: The new embedding vector (numpy array).
            alpha: The weighting factor for the moving average (0 < alpha <= 1).
                If alpha=1, the old embedding is completely replaced.
        """
        if not isinstance(person_id, str) or not person_id:
            logger.error("Invalid person_id provided (must be non-empty string).")
            return
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
            logger.error(f"Invalid embedding format for {person_id}. Must be a 1D numpy array.")
            return

        emb = embedding.astype(np.float32) # Ensure correct type

        if person_id in self.db:
            old_emb = self.db[person_id]
            # Ensure alpha is within a reasonable range
            alpha = np.clip(alpha, 0.0, 1.0)
            updated_emb = (1.0 - alpha) * old_emb + alpha * emb
            # Normalize the updated embedding
            norm = np.linalg.norm(updated_emb)
            self.db[person_id] = updated_emb / norm if norm > 1e-6 else updated_emb
            logger.debug(f"Updated embedding for '{person_id}' using EMA (alpha={alpha:.2f}).")
        else:
            # Normalize the initial embedding
            norm = np.linalg.norm(emb)
            self.db[person_id] = emb / norm if norm > 1e-6 else emb
            logger.info(f"Added new embedding for '{person_id}'.")

        self.dirty = True

    def get_database(self) -> Dict[str, np.ndarray]:
        """Returns the current in-memory database."""
        return self.db

    def get_embedding(self, person_id: str) -> Optional[np.ndarray]:
        """Returns the embedding for a specific person_id, or None if not found."""
        return self.db.get(person_id)

    def save_to_file(self, filename: Optional[str] = None) -> bool:
        """
        Saves the current database to a compressed .npz file.

        Args:
            filename: The path to save the file. Defaults to self.db_path.

        Returns:
            True if saving was successful, False otherwise.
        """
        save_path = filename or self.db_path
        try:
            # Ensure directory exists before saving
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            np.savez_compressed(save_path, **self.db)
            self.dirty = False # Reset dirty flag only on successful save
            logger.info(f"Database successfully saved to {save_path} with {len(self.db)} entries.")
            return True
        except IOError as e:
            logger.error(f"IOError saving database to {save_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error saving database to {save_path}: {e}", exc_info=True)
        return False

    def save_if_dirty(self, filename: Optional[str] = None) -> bool:
        """Saves the database only if changes have been made since the last save/load."""
        if self.dirty:
            logger.info("Database has unsaved changes. Saving...")
            return self.save_to_file(filename)
        else:
            logger.debug("Database save skipped: No changes detected.")
            return True # Considered successful as no save was needed

    def load_from_file(self, filename: Optional[str] = None) -> bool:
        """
        Loads the database from a .npz file, replacing the current in-memory db.

        Args:
            filename: The path to load the file from. Defaults to self.db_path.

        Returns:
            True if loading was successful, False otherwise.
        """
        load_path = filename or self.db_path
        if not os.path.exists(load_path):
            logger.warning(f"Database file not found: {load_path}. Starting with an empty database.")
            self.db = {}
            self.dirty = False # Loaded state is clean
            return True # Not an error, just no file to load

        try:
            loaded_data = np.load(load_path, allow_pickle=False)
            # Validate loaded data - ensure values are numpy arrays (basic check)
            self.db = {k: v for k, v in loaded_data.items() if isinstance(v, np.ndarray)}
            invalid_items = len(loaded_data) - len(self.db)
            if invalid_items > 0:
                logger.warning(f"Excluded {invalid_items} non-numpy array items during DB load.")

            self.dirty = False # Loaded state is clean
            logger.info(f"Database loaded successfully from {load_path} with {len(self.db)} entries.")
            return True
        except FileNotFoundError:
            logger.warning(f"Database file not found during load attempt: {load_path}. Starting fresh.")
            self.db = {}
            self.dirty = False
            return True # Return True as it's a defined state (empty db)
        except Exception as e:
            logger.error(f"Failed to load database from {load_path}: {e}", exc_info=True)
            self.db = {} # Reset to empty state on critical load error
            self.dirty = False
            return False