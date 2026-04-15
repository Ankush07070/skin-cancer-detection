import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image


import base64
import cv2
import numpy as np
from io import BytesIO
from .model_loader import load_model

sys.path.append(os.path.dirname(__file__))

from image_utils import process_image

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# =========================
# CONFIG
# =========================
MODEL_PATH = "artifacts/model_epoch_2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = load_model(MODEL_PATH, device)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)

# Label mapping
classes = [
    "Melanocytic Nevi",
    "Melanoma",
    "Benign Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratoses",
    "Vascular Lesions",
    "Dermatofibroma"
]
@app.route("/")
def home():
    return "🚀 Skin Cancer Detection API is running!"



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    input_tensor = process_image(image).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    # ===== Grad-CAM =====
    from src.components.gradcam import GradCAM

    target_layer = model.cnn[-1]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)

    # Convert original image
    img = np.array(image.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', overlay.astype("uint8"))
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "prediction": classes[pred.item()],
        "confidence": float(confidence.item()),
        "heatmap": heatmap_base64
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)