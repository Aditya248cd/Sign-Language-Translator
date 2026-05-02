import cv2
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.camera_utils import open_camera, read_frame
from src.sign_classes import DEFAULT_SIGN_CLASSES

classes = DEFAULT_SIGN_CLASSES
data_path = os.path.join(_ROOT, "data", "raw")
samples_per_class = 300

for class_name in classes:
    class_dir = os.path.join(data_path, class_name)
    os.makedirs(class_dir, exist_ok=True)

print("Opening webcam...")
cap, cam_index = open_camera()
if cap is None:
    print(
        "Error: Could not open any camera. Try:\n"
        "  • Close other apps using the camera (Teams, Zoom, browser /live tab).\n"
        "  • Set CAMERA_INDEX=1 (or 2) before running if you use an external webcam.\n"
        "  • On Windows, allow camera access in Settings → Privacy → Camera."
    )
    exit(1)

print(f"Webcam opened (camera index {cam_index}).")

for class_name in classes:
    print(f"\nCollecting data for: {class_name}")
    print("Press 's' to start capturing")

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            print("Can't grab a frame — check camera permissions / close other apps. Retrying…")
            continue

        display = frame.copy()
        cv2.putText(display, f"Class: {class_name} | Press 's' to start", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Stopped by user.")
            exit()

    count = 0
    while count < samples_per_class:
        ret, frame = read_frame(cap)
        if not ret:
            print("Can't grab a frame — retrying…")
            continue

        img_path = os.path.join(data_path, class_name, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        display = frame.copy()
        cv2.putText(display, f"{class_name}: {count}/{samples_per_class}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")