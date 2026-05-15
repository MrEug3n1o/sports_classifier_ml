import os
import io
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "sports-ball-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

# СТРОГОЕ ИСПРАВЛЕНИЕ: Имена классов ровно так, как их считал и отсортировал Colab (с пробелами)
CLASS_NAMES = [
    "american football",
    "baseball",
    "basketball",
    "billiard ball",
    "bowling ball",
    "cricket ball",
    "football",
    "golf ball",
    "hockey ball",
    "hockey puck",
    "rugby ball",
    "shuttlecock",
    "table tennis ball",
    "tennis ball",
    "volleyball",
]

CLASS_DISPLAY = {
    "american football": "American Football",
    "baseball": "Baseball",
    "basketball": "Basketball",
    "billiard ball": "Billiard Ball",
    "bowling ball": "Bowling Ball",
    "cricket ball": "Cricket Ball",
    "football": "Football (Soccer)",
    "golf ball": "Golf Ball",
    "hockey ball": "Field Hockey Ball",
    "hockey puck": "Hockey Puck",
    "rugby ball": "Rugby Ball",
    "shuttlecock": "Shuttlecock",
    "table_tennis_ball": "Table Tennis Ball", # Совместимость с формой вывода
    "table tennis ball": "Table Tennis Ball",
    "tennis ball": "Tennis Ball",
    "volleyball": "Volleyball",
}

CLASS_EMOJI = {
    "american football": "🏈",
    "baseball": "⚾",
    "basketball": "🏀",
    "billiard ball": "🎱",
    "bowling_ball": "🎳", # Совместимость
    "bowling ball": "🎳",
    "cricket ball": "🏏",
    "football": "⚽",
    "golf ball": "⛳",
    "hockey ball": "🏑",
    "hockey puck": "🏒",
    "rugby ball": "🏉",
    "shuttlecock": "🏸",
    "table tennis ball": "🏓",
    "tennis ball": "🎾",
    "volleyball": "🏐",
}

# Lazy-load the model
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            import tensorflow as tf

            # Путь к вашей финальной модели, сохраненной на шаге 11 ноутбука
            model_path = os.path.join(os.path.dirname(__file__), "model", "sports_ball_model.h5")
            if os.path.exists(model_path):
                _model = tf.keras.models.load_model(model_path)
                print("✅ Model loaded from", model_path)
            else:
                print("⚠️  Model file not found at", model_path)
                _model = None
        except ImportError:
            print("⚠️  TensorFlow not installed. Using mock predictions.")
            _model = None
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    # ИСПРАВЛЕНИЕ: Убрали деление на 255.0, так как EfficientNetB0 ожидает сырые пиксели 0-255!
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image_bytes):
    model = get_model()

    if model is None:
        # Demo mode: return a plausible random prediction
        import random
        idx = random.randint(0, len(CLASS_NAMES) - 1)
        confidence = random.uniform(0.60, 0.98)
        probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.3)
        probs[idx] = confidence
        probs /= probs.sum()
        top5_idx = np.argsort(probs)[::-1][:5]
        return {
            "class": CLASS_NAMES[idx],
            "display": CLASS_DISPLAY.get(CLASS_NAMES[idx], CLASS_NAMES[idx]),
            "emoji": CLASS_EMOJI.get(CLASS_NAMES[idx], "⚽"),
            "confidence": float(confidence * 100),
            "top5": [
                {
                    "class": CLASS_NAMES[i],
                    "display": CLASS_DISPLAY.get(CLASS_NAMES[i], CLASS_NAMES[i]),
                    "emoji": CLASS_EMOJI.get(CLASS_NAMES[i], "⚽"),
                    "confidence": float(probs[i] * 100),
                }
                for i in top5_idx
            ],
            "demo_mode": True,
        }

    arr = preprocess_image(image_bytes)
    preds = model.predict(arr)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    best_idx = top5_idx[0]

    return {
        "class": CLASS_NAMES[best_idx],
        "display": CLASS_DISPLAY.get(CLASS_NAMES[best_idx], CLASS_NAMES[best_idx]),
        "emoji": CLASS_EMOJI.get(CLASS_NAMES[best_idx], "⚽"),
        "confidence": float(preds[best_idx] * 100),
        "top5": [
            {
                "class": CLASS_NAMES[i],
                "display": CLASS_DISPLAY.get(CLASS_NAMES[i], CLASS_NAMES[i]),
                "emoji": CLASS_EMOJI.get(CLASS_NAMES[i], "⚽"),
                "confidence": float(preds[i] * 100),
            }
            for i in top5_idx
        ],
        "demo_mode": False,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    image_bytes = file.read()

    # Convert to base64 for passing to result page
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = file.filename.rsplit(".", 1)[1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    result = predict(image_bytes)

    return render_template(
        "result.html",
        result=result,
        image_data=f"data:{mime};base64,{img_b64}",
    )


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """JSON API endpoint for classification."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    result = predict(file.read())
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
