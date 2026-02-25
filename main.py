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
# Threshold badha kar 0.40 kar diya hai strict security ke liye
MATCH_THRESHOLD = 0.40 

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
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None

# --- NAYA SECURITY FUNCTION: PASSIVE LIVENESS CHECK ---
def check_liveness(image, face_box):
    """
    Check if the face is a real 3D person or a flat 2D photo/screen.
    Uses mathematical texture analysis (Laplacian Variance & Std Deviation).
    """
    x, y, w, h = [int(v) for v in face_box[:4]]
    img_h, img_w = image.shape[:2]
    
    # Extract the face region safely
    y1, y2 = max(0, y), min(img_h, y+h)
    x1, x2 = max(0, x), min(img_w, x+w)
    face_roi = image[y1:y2, x1:x2]

    if face_roi.size == 0:
        return False, "Invalid Face Box"

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # 1. Blur Detection (Printed photos tend to be blurry/flat)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Glare/Screen Detection (Screens emit flat light, reducing contrast variance)
    std_dev = np.std(gray)

    # Thresholds: Isse kam score aaya toh wo screen ya photo hai
    if lap_var < 35.0:
        return False, "Spoof Detected: Printed Photo (Low Sharpness)"
    
    if std_dev < 15.0:
        return False, "Spoof Detected: Digital Screen (Low Contrast)"

    return True, "Real Face"

def get_face_feature_and_liveness(image, check_fake=False):
    if image is None: return None, None, False, "No Image"

    height, width, _ = image.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    
    if faces is None:
        return None, None, False, "No face found"

    # Agar check_fake True hai, toh hum photo/screen check karenge
    if check_fake:
        is_real, spoof_msg = check_liveness(image, faces[0])
        if not is_real:
            return None, faces[0], False, spoof_msg # Reject as fake

    aligned_face = recognizer.alignCrop(image, faces[0])
    feature = recognizer.feature(aligned_face)
    return feature, faces[0], True, "Valid"

# --- 3. API ROUTES ---
@app.route('/')
def home():
    return jsonify({"status": "running", "model": "OpenCV SFace + Anti-Spoofing"})

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        data = request.json
        if not data or 'captured' not in data or 'master' not in data:
            return jsonify({"success": False, "error": "Missing image data"}), 400

        img_cap = base64_to_image(data['captured'])
        img_mst = base64_to_image(data['master'])

        if img_cap is None or img_mst is None:
            return jsonify({"success": False, "error": "Invalid Base64 Image"}), 400

        # Master image se feature nikalo (Isme spoof check ki zarurat nahi)
        feat_mst, _, _, _ = get_face_feature_and_liveness(img_mst, check_fake=False)
        if feat_mst is None:
            return jsonify({"success": False, "error": "No face found in 'master' image"}), 400

        # Captured live image se feature nikalo AUR spoof check karo
        feat_cap, _, is_real, spoof_msg = get_face_feature_and_liveness(img_cap, check_fake=True)
        
        # Agar spoof pakda gaya (photo ya screen)
        if not is_real:
            # Hum "match: False" bhejenge taaki frontend par "❌ Face Not Matched" likha aaye
            # Employee ko pata bhi nahi chalega ki wo block ho chuka hai fake photo ki wajah se.
            print(f"SECURITY ALERT: {spoof_msg}")
            return jsonify({
                "success": True, 
                "match": False, 
                "score": 0.0, 
                "error": spoof_msg,
                "threshold": MATCH_THRESHOLD
            })

        if feat_cap is None:
            return jsonify({"success": False, "error": "No face found in 'captured' image"}), 400

        # Agar asli chehra hai, tab compare karo
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
