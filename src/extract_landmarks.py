import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

# Allow running as script from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features import hand_landmarks_to_vector_normalized

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

data = []
labels = []

raw_path = os.path.join(_ROOT, "data", "raw")
processed_path = os.path.join(_ROOT, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

classes = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]

for class_name in classes:
    class_dir = os.path.join(raw_path, class_name)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            vec = hand_landmarks_to_vector_normalized(hand_landmarks)
            data.append(vec)
            labels.append(class_name)

X = np.array(data, dtype=np.float64)
y = np.array(labels)

np.save(os.path.join(processed_path, "X.npy"), X)
np.save(os.path.join(processed_path, "y.npy"), y)

meta_path = os.path.join(processed_path, "feature_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({"feature_mode": "normalized"}, f, indent=2)

print("Landmark extraction completed (wrist-normalized features).")
print("X shape:", X.shape)
print("y shape:", y.shape)
