"""
Desktop live preview with hand box and sign label (OpenCV window).
Requires a trained model in models/. Run from project root: python camera_test.py
"""
import os
import sys

import cv2

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.camera_utils import open_camera, read_frame
from src.predict_image import get_predictor


def main() -> None:
    cap, idx = open_camera()
    if cap is None:
        print(
            "Camera not opening. Close other apps using the camera or set CAMERA_INDEX=1"
        )
        return
    print(f"Using camera index {idx} — press Q to quit")

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        print(e)
        cap.release()
        return

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            print("Can't grab a frame — check camera; retrying…")
            continue

        out = predictor.predict_bgr(frame)
        disp = frame.copy()
        label = out.get("label")
        err = out.get("error")
        bbox = out.get("bbox")

        if bbox and label and not err:
            x, y = int(bbox["x"]), int(bbox["y"])
            bw, bh = int(bbox["width"]), int(bbox["height"])
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), (253, 139, 61), 2, cv2.LINE_AA)
            conf = out.get("confidence")
            text = f"{label}"
            if conf is not None:
                text = f"{label} {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            ty = y - 8 if y - 8 > th else y + bh + th + 4
            tx = x
            cv2.rectangle(
                disp,
                (tx, ty - th - 8),
                (tx + tw + 12, ty + 4),
                (30, 35, 45),
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                disp,
                text,
                (tx + 6, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (232, 238, 244),
                2,
                cv2.LINE_AA,
            )
        elif err:
            cv2.putText(
                disp,
                str(err),
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (120, 120, 120),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Sign language — live", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
