from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import trained ML models
from cloud_removal import Generator  # CycleGAN Model
from coco_segmentation import UNet, Config as SegConfig  # U-Net Model

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "static/results")
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")  # User's Downloads folder
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model Checkpoints
CLOUD_REMOVAL_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "cycleGAN_checkpoint.pth")
SEGMENTATION_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "unet_final.pth")

# Ensure Checkpoints Exist
if not os.path.exists(CLOUD_REMOVAL_CHECKPOINT):
    raise FileNotFoundError(f"Cloud Removal model checkpoint not found at {CLOUD_REMOVAL_CHECKPOINT}")

if not os.path.exists(SEGMENTATION_CHECKPOINT):
    raise FileNotFoundError(f"Segmentation model checkpoint not found at {SEGMENTATION_CHECKPOINT}")

# Load PyTorch models on the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Cloud Removal Model (CycleGAN)
cloud_removal_model = Generator().to(device)
checkpoint = torch.load(CLOUD_REMOVAL_CHECKPOINT, map_location=device)
cloud_removal_model.load_state_dict(checkpoint["G_cloudy2clear"])
cloud_removal_model.eval()

# Load Land Segmentation Model (U-Net)
segmentation_model = UNet(n_channels=3, n_classes=SegConfig.NUM_CLASSES).to(device)
checkpoint = torch.load(SEGMENTATION_CHECKPOINT, map_location=device)
segmentation_model.load_state_dict(checkpoint["model_state_dict"])
segmentation_model.eval()

# Image Processing Functions
def preprocess_image(image, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    return transforms.ToPILImage()(tensor)

# ✅ Home Route (Render HTML)
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Cloud Removal API
@app.route("/remove-clouds/", methods=["POST"])
def remove_clouds():
    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output_tensor = cloud_removal_model(input_tensor)

        output_image = postprocess_image(output_tensor)

        filename = f"cloud_removed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = os.path.join(RESULTS_DIR, filename)
        output_image.save(output_path)

        # Automatically save to Downloads folder
        shutil.copy(output_path, os.path.join(DOWNLOADS_DIR, filename))

        return jsonify({"message": "Clouds removed successfully", "filename": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Land Segmentation API
@app.route("/analyze-image/", methods=["POST"])
def analyze_image():
    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = preprocess_image(image, size=(SegConfig.IMAGE_SIZE, SegConfig.IMAGE_SIZE))

        with torch.no_grad():
            output = segmentation_model(input_tensor)
            pred = output.argmax(dim=1)

        filename = f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = os.path.join(RESULTS_DIR, filename)
        plt.imsave(output_path, pred.squeeze().cpu().numpy(), cmap="jet")

        # Automatically save to Downloads folder
        shutil.copy(output_path, os.path.join(DOWNLOADS_DIR, filename))

        return jsonify({"filename": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Serve Processed Images
@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

# Run Flask App
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)