import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # फालतू logs बंद करने के लिए
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Face AI Server is Running!"

@app.route('/verify', methods=['POST'])
def verify_face():
    # Unique filenames taaki multiple users ka data mix na ho
    u_id = uuid.uuid4().hex
    cap_path = f"captured_{u_id}.jpg"
    mast_path = f"master_{u_id}.jpg"
    
    try:
        data = request.json
        img_captured_base64 = data['captured'].split(",")[1]
        img_master_base64 = data['master'].split(",")[1]

        # Base64 se images save karna
        with open(cap_path, "wb") as f:
            f.write(base64.b64decode(img_captured_base64))
        with open(mast_path, "wb") as f:
            f.write(base64.b64decode(img_master_base64))

       # Facenet512 ki jagah OpenFace use karein (RAM bachaane ke liye)
        result = DeepFace.verify(
            img1_path = cap_path, 
            img2_path = mast_path, 
            model_name = "OpenFace", # Ye 512MB RAM ke liye best hai
            distance_metric = "cosine",
            enforce_detection = False
        )

        # Cleanup: Files delete karna zaroori hai
        if os.path.exists(cap_path): os.remove(cap_path)
        if os.path.exists(mast_path): os.remove(mast_path)

        return jsonify({
            "success": True,
            "match": result["verified"],
            "distance": result["distance"]
        })

    except Exception as e:
        if os.path.exists(cap_path): os.remove(cap_path)
        if os.path.exists(mast_path): os.remove(mast_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Render ke liye port 10000 zaroori hota hai
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
