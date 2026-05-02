from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any, Callable

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image

from src.features import hand_landmarks_to_vector_normalized, hand_landmarks_to_vector_raw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLE_PATH = os.path.join(_PROJECT_ROOT, "models", "model_bundle.pkl")
_LEGACY_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "sign_model.pkl")
_LEGACY_ENCODER_PATH = os.path.join(_PROJECT_ROOT, "models", "label_encoder.pkl")


def hand_bbox_pixels(
    hand_landmarks, width: int, height: int, pad: float = 0.18
) -> dict[str, int]:
    """Tight box around all landmarks in pixel coords; pad in normalized space."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    px_pad = span * pad + 0.03
    min_x = max(0.0, min_x - px_pad)
    max_x = min(1.0, max_x + px_pad)
    min_y = max(0.0, min_y - px_pad)
    max_y = min(1.0, max_y + px_pad)
    x1 = int(min_x * width)
    y1 = int(min_y * height)
    x2 = int(max_x * width)
    y2 = int(max_y * height)
    return {
        "x": x1,
        "y": y1,
        "width": max(1, x2 - x1),
        "height": max(1, y2 - y1),
    }


def _top_k_probs(
    label_encoder, proba: np.ndarray, k: int = 3
) -> list[dict[str, Any]]:
    k = min(k, len(proba[0]))
    idx = np.argsort(-proba[0])[:k]
    return [
        {
            "label": str(label_encoder.inverse_transform([int(i)])[0]),
            "confidence": float(proba[0][i]),
        }
        for i in idx
    ]


class SignPredictor:
    """Loads sklearn model (pipeline or legacy estimator) + label encoder + feature fn."""

    def __init__(
        self,
        model: Any,
        label_encoder: Any,
        vectorize: Callable[..., np.ndarray],
    ):
        self._model = model
        self._label_encoder = label_encoder
        self._vectorize = vectorize
        self._hand_lock = threading.Lock()
        mp_hands = mp.solutions.hands
        # Each HTTP live frame is an independent image, not a continuous video stream.
        # static_image_mode=False expects strictly increasing timestamps and breaks
        # when Flask handles overlapping requests or out-of-order frames.
        self._hands_live = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._hands_live.close()

    def _predict_from_landmarks(self, hand_landmarks) -> dict[str, Any]:
        x = self._vectorize(hand_landmarks).reshape(1, -1)
        pred = self._model.predict(x)
        label = str(self._label_encoder.inverse_transform(pred)[0])
        out: dict[str, Any] = {"label": label, "confidence": None, "top3": []}
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(x)
            out["confidence"] = float(np.max(proba[0]))
            out["top3"] = _top_k_probs(self._label_encoder, proba, 3)
        return out

    def predict_bgr(self, bgr: np.ndarray) -> dict[str, Any]:
        if bgr is None or bgr.size == 0:
            return {"error": "Invalid image", "label": None}
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._hand_lock:
            result = self._hands_live.process(rgb)
        if not result.multi_hand_landmarks:
            return {"error": "No hand detected", "label": None}
        hand = result.multi_hand_landmarks[0]
        pred = self._predict_from_landmarks(hand)
        pred["bbox"] = hand_bbox_pixels(hand, w, h)
        pred["image_width"] = int(w)
        pred["image_height"] = int(h)
        pred["error"] = None
        return pred

    def predict_image_path(self, image_path: str) -> dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Invalid image", "label": None}
        ih, iw = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_hands = mp.solutions.hands
        hands_still = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
        )
        try:
            result = hands_still.process(rgb)
        finally:
            hands_still.close()
        if not result.multi_hand_landmarks:
            return {"error": "No hand detected", "label": None}
        hand = result.multi_hand_landmarks[0]
        pred = self._predict_from_landmarks(hand)
        pred["bbox"] = hand_bbox_pixels(hand, iw, ih)
        pred["image_width"] = int(iw)
        pred["image_height"] = int(ih)
        pred["error"] = None
        return pred


_predictor: SignPredictor | None = None


def get_predictor() -> SignPredictor:
    global _predictor
    if _predictor is not None:
        return _predictor

    if os.path.isfile(_BUNDLE_PATH):
        bundle = joblib.load(_BUNDLE_PATH)
        mode = bundle.get("feature_mode", "raw")
        vec = (
            hand_landmarks_to_vector_normalized
            if mode == "normalized"
            else hand_landmarks_to_vector_raw
        )
        _predictor = SignPredictor(bundle["model"], bundle["label_encoder"], vec)
        return _predictor

    if os.path.isfile(_LEGACY_MODEL_PATH) and os.path.isfile(_LEGACY_ENCODER_PATH):
        model = joblib.load(_LEGACY_MODEL_PATH)
        le = joblib.load(_LEGACY_ENCODER_PATH)
        _predictor = SignPredictor(model, le, hand_landmarks_to_vector_raw)
        return _predictor

    raise FileNotFoundError(
        "No trained model found. Expected models/model_bundle.pkl "
        "or models/sign_model.pkl + models/label_encoder.pkl. "
        "Run: python src/extract_landmarks.py && python src/train_model.py"
    )


def predict_sign(image_path: str) -> str:
    """Backward-compatible string result for templates."""
    try:
        p = get_predictor().predict_image_path(image_path)
    except FileNotFoundError as e:
        return str(e)
    if p.get("error"):
        return p["error"]
    conf = p.get("confidence")
    if conf is not None:
        return f"{p['label']} ({conf * 100:.1f}% confidence)"
    return str(p["label"])


def predict_sign_detail(image_path: str) -> dict[str, Any]:
    try:
        return get_predictor().predict_image_path(image_path)
    except FileNotFoundError as e:
        return {"error": str(e), "label": None}


def predict_from_base64_jpeg(
    data_url_or_b64: str, *, max_width: int = 480
) -> dict[str, Any]:
    """Decode a data URL or raw base64 JPEG and run live-style prediction."""
    s = data_url_or_b64.strip()
    if "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ih, iw = bgr.shape[:2]
    if iw > max_width:
        scale = max_width / iw
        nh = max(1, int(round(ih * scale)))
        bgr = cv2.resize(bgr, (max_width, nh), interpolation=cv2.INTER_AREA)
    return get_predictor().predict_bgr(bgr)
