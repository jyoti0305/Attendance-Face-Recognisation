from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Taaki Apps Script se request block na ho

@app.route('/')
def home():
    return "Face API is Running!"

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        data = request.json
        # 1. Captured photo aur Master photo (jo Drive se aayegi) nikalna
        img_captured_base64 = data['captured'].split(",")[1]
        img_master_base64 = data['master'].split(",")[1]

        # 2. Temporary files mein save karna taaki DeepFace read kar sake
        with open("captured.jpg", "wb") as f:
            f.write(base64.b64decode(img_captured_base64))
        with open("master.jpg", "wb") as f:
            f.write(base64.b64decode(img_master_base64))

        # 3. DeepFace Match Logic
        # VGG-Face model kaafi accurate hai attendance ke liye
        result = DeepFace.verify(
            img1_path = "captured.jpg", 
            img2_path = "master.jpg", 
            model_name = "VGG-Face",
            distance_metric = "cosine",
            enforce_detection = True # Isse photo mein face hona zaroori hai
        )

        # 4. Result bhejna
        # threshold 0.40 se 0.50 ke beech rakhein (jitna kam, utna strict)
        is_matched = result["verified"]
        
        return jsonify({
            "success": True,
            "match": is_matched,
            "distance": result["distance"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)