import os
import re

from flask import Flask, jsonify, render_template, request, url_for

from src.predict_image import get_predictor, predict_from_base64_jpeg, predict_sign_detail

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"[^\w.\-]", "_", base)
    return base or "upload.jpg"


@app.route("/")
def home():
    return render_template(
        "index.html",
        page_title="Sign Language Translator | Home",
        active_page="home",
        body_class="page-home",
    )


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    prediction = None
    detail = None
    image_url = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename != "":
                fname = _safe_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                file.save(filepath)
                detail = predict_sign_detail(filepath)
                image_url = url_for("static", filename=f"uploads/{fname}")
                if detail.get("error"):
                    prediction = detail["error"]
                elif detail.get("label") is not None:
                    conf = detail.get("confidence")
                    prediction = (
                        f"{detail['label']} ({conf * 100:.1f}% confidence)"
                        if conf is not None
                        else detail["label"]
                    )

    return render_template(
        "dashboard_pro.html",
        prediction=prediction,
        detail=detail,
        image_url=image_url,
        page_title="Sign Language Translator | Upload",
        active_page="dashboard",
        body_class="page-dashboard",
    )


@app.route("/live")
def live():
    return render_template(
        "live_pro.html",
        page_title="Sign Language Translator | Live Studio",
        active_page="live",
        body_class="page-live",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    if not image_b64 or not isinstance(image_b64, str):
        return jsonify({"ok": False, "error": "Missing JSON field 'image' (base64 or data URL)."}), 400
    try:
        pred = predict_from_base64_jpeg(image_b64)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if pred.get("error"):
        return jsonify({"ok": True, "hand_detected": False, "message": pred["error"]})
    return jsonify(
        {
            "ok": True,
            "hand_detected": True,
            "label": pred.get("label"),
            "confidence": pred.get("confidence"),
            "top3": pred.get("top3") or [],
            "bbox": pred.get("bbox"),
            "image_width": pred.get("image_width"),
            "image_height": pred.get("image_height"),
        }
    )


@app.route("/about")
def about():
    return render_template(
        "about_pro.html",
        page_title="Sign Language Translator | About",
        active_page="about",
        body_class="page-about",
    )


if __name__ == "__main__":
    app.run(debug=True)
