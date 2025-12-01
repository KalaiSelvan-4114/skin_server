from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import datetime
import traceback
import io
import numpy as np
import cv2

app = Flask(__name__)

# Folder to save images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------- Load YOLO model once at startup -------------
# Put your trained weights file (from Colab) in same folder as this script
# e.g. "skin.pt" or "best.pt"
MODEL_PATH = os.environ.get("MODEL_PATH", "skin_model.pt")    # change via env if needed
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")  # 'cpu' or 'cuda'

print(f"[INFO] Loading model from {MODEL_PATH} on device {MODEL_DEVICE} ...")
try:
    model = YOLO(MODEL_PATH, device=MODEL_DEVICE)
    print("[INFO] Model loaded ✅")
except Exception as e:
    tb = traceback.format_exc()
    print("[ERROR] Failed to load model:")
    print(tb)
    # Try fallback to a small official model to keep service alive
    try:
        fallback = "yolov8n.pt"
        print(f"[INFO] Attempting to load fallback model {fallback} on cpu...")
        model = YOLO(fallback, device='cpu')
        print("[INFO] Fallback model loaded ✅ (cpu)")
    except Exception as e2:
        tb2 = traceback.format_exc()
        print("[CRITICAL] Fallback model load failed:")
        print(tb2)
        raise


@app.route("/upload", methods=["POST"])
def upload():
    # Raw JPEG bytes from ESP32-CAM
    img_bytes = request.get_data()

    if not img_bytes:
        return jsonify({"status": "error", "message": "No image data"}), 400

    # Create a filename with timestamp and save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{ts}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        print(f"[INFO] Saved: {filepath}, size = {len(img_bytes)} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to save image: {e}")
        return jsonify({"status": "error", "message": "Failed to save image"}), 500

    # ------------- Run YOLO inference -------------
    try:
        # Run model on the saved image
        results = model(filepath)[0]   # first (and only) image

        if results.boxes is None or len(results.boxes) == 0:
            # No detection
            result_label = "No disease"
            confidence_str = "0%"
        else:
            boxes = results.boxes

            # Pick the most confident detection
            best_idx = boxes.conf.argmax().item()
            cls_id = int(boxes.cls[best_idx].item())
            conf = float(boxes.conf[best_idx].item())

            result_label = model.names[cls_id]       # class name from YOLO
            confidence_str = f"{conf * 100:.1f}%"

        print(f"[INFO] Prediction: {result_label} ({confidence_str})")

    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] Inference failed:")
        print(tb)
        return jsonify({
            "status": "error",
            "message": "Model inference failed",
            "error": str(e)
        }), 500

    # ------------- Send result back to ESP32 -------------
    return jsonify({
        "status": "success",
        "image": filename,
        "result": result_label,
        "confidence": confidence_str
    }), 200


if __name__ == "__main__":
    # Run on all interfaces so ESP32 can reach it over LAN
    app.run(host="0.0.0.0", port=5000, debug=True)


@app.route("/health", methods=["GET"])
def health():
    try:
        loaded = model is not None
        names = getattr(model, "names", None)
        return jsonify({
            "status": "ok" if loaded else "error",
            "model_loaded": bool(loaded),
            "classes": len(names) if names is not None else None
        }), 200
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] Health check failed:\n", tb)
        return jsonify({"status": "error", "message": str(e)}), 500
