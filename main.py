import os
import cv2
import numpy as np
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Threshold: Isse upar score aaya to "Match" hai.
# SFace ke liye 0.363 standard cosine similarity threshold hai.
MATCH_THRESHOLD = 0.363 

# Folder jahan AI models save honge
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Model URLs (Official OpenCV Zoo)
MODELS = {
    "detector": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx"
    },
    "recognizer": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "filename": "face_recognition_sface_2021dec.onnx"
    }
}

# --- 1. GLOBAL MODEL LOADING ---
def get_model_path(model_key):
    path = os.path.join(WEIGHTS_DIR, MODELS[model_key]["filename"])
    if not os.path.exists(path):
        print(f"Downloading {model_key} model... please wait.")
        try:
            r = requests.get(MODELS[model_key]["url"], stream=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{model_key} downloaded successfully.")
        except Exception as e:
            print(f"Failed to download {model_key}: {e}")
            return None
    return path

# Initialize Models Globally (runs once at startup)
print("Initializing AI Models...")
detector_path = get_model_path("detector")
recognizer_path = get_model_path("recognizer")

if detector_path and recognizer_path:
    # YuNet for finding faces
    detector = cv2.FaceDetectorYN.create(
        detector_path, "", (320, 320), 0.9, 0.3, 5000
    )
    # SFace for recognizing faces
    recognizer = cv2.FaceRecognizerSF.create(
        recognizer_path, ""
    )
    print("Models Loaded & Ready!")
else:
    print("CRITICAL ERROR: Could not load models.")
    exit(1)

# --- 2. HELPER FUNCTIONS ---
def base64_to_image(b64_str):
    """Base64 string ko OpenCV image mein convert karta hai (Memory mein)."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None

def get_face_feature(image):
    """Image se face dhoondh kar uska 'feature vector' nikalta hai."""
    if image is None: return None

    height, width, _ = image.shape
    # Detector ko image size batana zaroori hai
    detector.setInputSize((width, height))

    # Face Detection
    # faces variable format: [x1, y1, w, h, x_re, y_re, ...]
    _, faces = detector.detect(image)
    
    # Agar face nahi mila
    if faces is None:
        return None

    # Sirf pehla face lete hain (faces[0])
    # Align and Crop (Face ko seedha karna)
    aligned_face = recognizer.alignCrop(image, faces[0])
    
    # Feature Extraction (128D vector)
    feature = recognizer.feature(aligned_face)
    return feature

# --- 3. API ROUTES ---
@app.route('/')
def home():
    return jsonify({"status": "running", "model": "OpenCV SFace"})

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        data = request.json
        if not data or 'captured' not in data or 'master' not in data:
            return jsonify({"success": False, "error": "Missing image data"}), 400

        # Images load karo
        img_cap = base64_to_image(data['captured'])
        img_mst = base64_to_image(data['master'])

        if img_cap is None or img_mst is None:
            return jsonify({"success": False, "error": "Invalid Base64 Image"}), 400

        # Features nikalo
        feat_cap = get_face_feature(img_cap)
        feat_mst = get_face_feature(img_mst)

        if feat_cap is None:
            return jsonify({"success": False, "error": "No face found in 'captured' image"}), 400
        if feat_mst is None:
            return jsonify({"success": False, "error": "No face found in 'master' image"}), 400

        # Compare karo (Cosine Similarity)
        # Result ek score hota hai. Higher is better match.
        score = recognizer.match(feat_cap, feat_mst, cv2.FaceRecognizerSF_FR_COSINE)
        
        is_match = bool(score >= MATCH_THRESHOLD)

        return jsonify({
            "success": True,
            "match": is_match,
            "score": float(score),
            "threshold": MATCH_THRESHOLD
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
