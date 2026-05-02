import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

processed = os.path.join(_ROOT, "data", "processed")
X = np.load(os.path.join(processed, "X.npy"))
y = np.load(os.path.join(processed, "y.npy"))

meta_path = os.path.join(processed, "feature_meta.json")
feature_mode = "raw"
if os.path.isfile(meta_path):
    with open(meta_path, encoding="utf-8") as f:
        feature_mode = json.load(f).get("feature_mode", "raw")
print("Feature mode from processed data:", feature_mode)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.08,
                max_iter=250,
                random_state=42,
                min_samples_leaf=5,
            ),
        ),
    ]
)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(
    classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
)

models_dir = os.path.join(_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)

bundle = {
    "model": pipeline,
    "label_encoder": label_encoder,
    "feature_mode": feature_mode,
}
joblib.dump(bundle, os.path.join(models_dir, "model_bundle.pkl"))

print("Saved models/model_bundle.pkl (retrain after extract_landmarks.py).")
