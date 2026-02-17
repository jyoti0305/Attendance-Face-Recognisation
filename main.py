import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import uuid

app = Flask(__name__)
# CORS allow karna zaroori hai taaki Google Apps Script se request aaye
CORS(app)

@app.route('/')
def home():
    return "Face AI Server is Running! OpenFace Model Active."

@app.route('/verify', methods=['POST'])
def verify_face():
    u_id = uuid.uuid4().hex
    cap_path = f"captured_{u_id}.jpg"
    mast_path = f"master_{u_id}.jpg"
    
    try:
        data = request.json
        if not data or 'captured' not in data or 'master' not in data:
            return jsonify({"success": False, "error": "Missing image data"}), 400

        # Base64 string se image data nikalna
        img_captured_base64 = data['captured'].split(",")[1]
        img_master_base64 = data['master'].split(",")[1]

        # Temporary files save karna
        with open(cap_path, "wb") as f:
            f.write(base64.b64decode(img_captured_base64))
        with open(mast_path, "wb") as f:
            f.write(base64.b64decode(img_master_base64))

        # --- DEEPFACE CORE LOGIC ---
        # model_name="OpenFace" speed ke liye best hai
        # detector_backend="opencv" fast hai aur kam memory leta hai
        result = DeepFace.verify(
            img1_path = cap_path, 
            img2_path = mast_path, 
            model_name = "OpenFace", 
            distance_metric = "cosine",
            enforce_detection = False, # Face detect na bhi ho tab bhi process kare
            detector_backend = "opencv" 
        )

        # Distance check logic
        # 0.55 se badha kar 0.60 ya 0.65 kar sakte hain agar matching hard ho rahi ho
        # OpenFace ke liye cosine distance 0.40 - 0.60 ke beech ideal hai
        is_matched = bool(result["distance"] < 0.60) 

        # Cleanup: Files ko turant delete karein taaki storage na bhare
        if os.path.exists(cap_path): os.remove(cap_path)
        if os.path.exists(mast_path): os.remove(mast_path)

        return jsonify({
            "success": True,
            "match": is_matched,
            "distance": float(result["distance"]),
            "model": "OpenFace"
        })

    except Exception as e:
        # Error aane par bhi files remove karein
        if os.path.exists(cap_path): os.remove(cap_path)
        if os.path.exists(mast_path): os.remove(mast_path)
        print(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Render default port 10000 use karta hai
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
