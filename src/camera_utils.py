"""Reliable webcam open + frame read (especially on Windows)."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional, Tuple

import cv2


def _try_open(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    # Warm the pipeline — first reads often fail on DSHOW/MSMF.
    for _ in range(20):
        cap.grab()
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            return cap
        time.sleep(0.03)
    cap.release()
    return None


def open_camera(index: Optional[int] = None) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """
    Open a working capture. Tries DirectShow on Windows first, then MSMF, then default.
    If index is None, reads env CAMERA_INDEX or tries 0..3.
    Returns (cap, index_used) or (None, None).
    """
    if index is None:
        env = os.environ.get("CAMERA_INDEX", "").strip()
        indices = [int(env)] if env.isdigit() else [0, 1, 2, 3]
    else:
        indices = [index]

    win = sys.platform == "win32"
    for idx in indices:
        if win:
            cap = _try_open(idx, cv2.CAP_DSHOW)
            if cap is not None:
                return cap, idx
            if hasattr(cv2, "CAP_MSMF"):
                cap = _try_open(idx, cv2.CAP_MSMF)
                if cap is not None:
                    return cap, idx
        cap = _try_open(idx, cv2.CAP_ANY)
        if cap is not None:
            return cap, idx
    return None, None


def read_frame(cap: cv2.VideoCapture, retries: int = 25) -> Tuple[bool, Optional[Any]]:
    """Read with retries (camera busy or transient empty buffer)."""
    for _ in range(retries):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(0.02)
    return False, None
