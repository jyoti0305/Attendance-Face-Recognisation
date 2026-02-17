import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Face AI Server is Running!"

@app.route('/verify', methods=['POST'])
def verify_face():
    u_id = uuid.uuid4().hex
    cap_path = f"captured_{u_id}.jpg"
    mast_path = f"master_{u_id}.jpg"
    
    try:
        data = request.json
        img_captured_base64 = data['captured'].split(",")[1]
        img_master_base64 = data['master'].split(",")[1]

        with open(cap_path, "wb") as f:
            f.write(base64.b64decode(img_captured_base64))
        with open(mast_path, "wb") as f:
            f.write(base64.b64decode(img_master_base64))

        # DeepFace Verification logic (Indentation Fixed)
        result = DeepFace.verify(
            img1_path = cap_path, 
            img2_path = mast_path, 
            model_name = "OpenFace", 
            distance_metric = "cosine",
            enforce_detection = False
        )

        # Distance check logic (Match threshold 0.55)
        is_matched = result["distance"] < 0.55

        # Cleanup
        if os.path.exists(cap_path): os.remove(cap_path)
        if os.path.exists(mast_path): os.remove(mast_path)

        return jsonify({
            "success": True,
            "match": is_matched,
            "distance": result["distance"]
        })

    except Exception as e:
        if os.path.exists(cap_path): os.remove(cap_path)
        if os.path.exists(mast_path): os.remove(mast_path)
        print(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
