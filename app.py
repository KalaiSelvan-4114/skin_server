from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import datetime

app = Flask(__name__)

# Folder to save images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------- Load YOLO model once at startup -------------
# Put your trained weights file (from Colab) in same folder as this script
# e.g. "skin.pt" or "best.pt"
MODEL_PATH = "skin_model.pt"    # change if your file name is different
print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded ✅")


@app.route("/upload", methods=["POST"])
def upload():
    # Raw JPEG bytes from ESP32-CAM
    img_bytes = request.get_data()

    if not img_bytes:
        return jsonify({"status": "error", "message": "No image data"}), 400

    # Create a filename with timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{ts}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save image to disk
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    print(f"[INFO] Saved: {filepath}, size = {len(img_bytes)} bytes")

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
        print("[ERROR] Inference failed:", e)
        return jsonify({
            "status": "error",
            "message": "Model inference failed"
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
