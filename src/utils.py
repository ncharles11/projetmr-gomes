#src/utils.py
import cv2
import numpy as np
import logging
from src import config
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def check_image_quality(face_image: np.ndarray, blur_limit: float = config.BLUR_THRESHOLD) -> bool:
    """
    Checks if the face image quality (blur) is sufficient.

    Args:
        face_image: The aligned face image (expected BGR).
        blur_limit: The minimum Laplacian variance threshold.

    Returns:
        True if quality is sufficient, False otherwise.
    """
    if face_image is None or face_image.size == 0:
        return False
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.debug(f"Image blur variance: {blur_variance:.2f}")
        is_sharp_enough = blur_variance > blur_limit
        if not is_sharp_enough:
             logger.warning(f"Image quality below threshold (Blur: {blur_variance:.2f} < {blur_limit:.2f})")
        return is_sharp_enough
    except cv2.error as e:
        logger.error(f"OpenCV error during quality check: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during quality check: {e}", exc_info=True)
        return False
    
def draw_detection(frame: np.ndarray, box: np.ndarray, landmark: Optional[np.ndarray], label: Optional[str] = None, color: Tuple[int,int,int]=(0, 255, 0)):
    """Draws bounding box, landmarks, and label on the frame."""
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if landmark is not None:
        for lx, ly in landmark.astype(int):
            cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1) # Red landmarks

    if label:
        # Put text slightly above the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_label = max(y1 - 10, label_size[1] + 5) # Ensure label is within frame boundaries
        # Simple background rectangle for label
        cv2.rectangle(frame, (x1, y1_label - label_size[1] - base_line),
                      (x1 + label_size[0], y1_label + base_line), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, y1_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black text

def estimate_pose(frame: np.ndarray,
                  box:   np.ndarray,
                  lmks:  np.ndarray
                 ):
    """
    Return (is_valid, info_dict) where `is_valid` is True when **all**
    of the following are satisfied:
        • face fully inside frame  (completeness >= MIN_COMP)
        • face big enough          (size_valid)
        • yaw / pitch / roll ≤ MAX_POSE_DEG

    `info_dict` contains the individual flags and the raw angles.
    """
    h,  w  = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)

    # --- completeness ---
    margin   = 15 # px guard band
    inside   = (x1 >= margin and y1 >= margin and
                x2 <= w - margin and y2 <= h - margin)

    inter_w  = max(0, min(x2, w) - max(x1, 0))
    inter_h  = max(0, min(y2, h) - max(y1, 0))
    comp     = (inter_w * inter_h) / float((x2 - x1) * (y2 - y1) + 1e-6)

    # --- size gate ---
    face_h      = (y2 - y1)
    face_frac   = face_h / float(h) # height-based criterion
    size_valid  = face_frac >= config.MIN_FACE_FRAC

    # --- landmarks ----
    re, le, nose, mr, ml = lmks.astype(float) # detector order
    LE, RE = le, re                           # swap eyes
    LM, RM = ml, mr                           # swap mouth corners

    # extra safety if frame might be mirrored
    if RE[0] < LE[0]:
        LE, RE = RE, LE
    if RM[0] < LM[0]:
        LM, RM = RM, LM

    img_pts = np.stack([nose, LE, RE, LM, RM]).astype(np.float64)

    # --- camera matrix ---
    f = 0.5 * (w + h)  # empirical focal length
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0,   1  ]], dtype=np.float64)
    
    MODEL_POINTS = np.array([
        (   0.,    0.,    0.),      # nose tip
        (-225., -170., -135.),      # left eye
        ( 225., -170., -135.),      # right eye
        (-150.,  150., -125.),      # left mouth
        ( 150.,  150., -125.)       # right mouth
    ], dtype=np.float64)

    succ, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_pts, K, None,
                                    flags=cv2.SOLVEPNP_EPNP)
    if not succ:
        return False, {'error': 'PnP failed'}

    pitch, yaw_raw, roll = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]

    # Apply offsets to yaw to match the configuration
    yaw = yaw_raw - config.YAW_OFFSET_DEG if yaw_raw >= 0 else yaw_raw + config.YAW_OFFSET_DEG

    # --- pose gate ---
    pose_valid = (abs(yaw)   <= config.MAX_POSE_DEG and
                  abs(pitch) <= config.MAX_POSE_DEG and
                  abs(roll)  <= config.MAX_POSE_DEG)

    # --- summary ---
    is_valid = inside and (comp >= config.MIN_COMPLETENESS) and size_valid and pose_valid

    info = dict(
        yaw=yaw, pitch=pitch, roll=roll,
        completeness=comp,   face_complete=inside,
        pose_valid=pose_valid,
        face_frac=face_frac, size_valid=size_valid
    )
    return is_valid, info