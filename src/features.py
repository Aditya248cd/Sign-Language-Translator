"""Hand landmark feature vectors shared by training, image predict, and live webcam."""

from __future__ import annotations

import numpy as np


def hand_landmarks_to_vector_normalized(hand_landmarks) -> np.ndarray:
    """
    Wrist-centered, scale-normalized 21x3 landmarks (63 floats).
    Reduces sensitivity to hand position and size in the frame.
    """
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )
    wrist = pts[0]
    rel = pts - wrist
    scale = float(np.linalg.norm(rel, axis=1).max()) + 1e-6
    return (rel / scale).flatten()


def hand_landmarks_to_vector_raw(hand_landmarks) -> np.ndarray:
    """Legacy flat vector (63 floats) in image-normalized MediaPipe coordinates."""
    return np.array(
        [c for lm in hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)],
        dtype=np.float64,
    )
