import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from PIL import Image

from app.model_loader import load_model
from app.image_utils import process_image

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# CONFIG
# =========================
MODEL_PATH = "artifacts/model_epoch_2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
try:
    model = load_model(MODEL_PATH, device)
    model.eval()
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)

# =========================
# CLASS LABELS
# =========================
classes = [
    "Melanocytic Nevi",
    "Melanoma",
    "Benign Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratoses",
    "Vascular Lesions",
    "Dermatofibroma"
]

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "🚀 Skin Cancer Detection API is running!"

# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check file
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        # Read image
        image = Image.open(file).convert("RGB")

        # Preprocess image
        input_tensor = process_image(image).to(device)

        # Prediction
        with torch.no_grad():

            outputs = model(input_tensor)

            probs = torch.softmax(outputs, dim=1)

            confidence, pred = torch.max(probs, 1)

        prediction = classes[pred.item()]

        confidence_score = float(confidence.item())

        # Return response
        return jsonify({
            "prediction": prediction,
            "confidence": confidence_score
        })

    except Exception as e:

        print("❌ Prediction Error:", e)

        return jsonify({
            "error": str(e)
        }), 500

# =========================
# RUN APP
# =========================
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=True
    )