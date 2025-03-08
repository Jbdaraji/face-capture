import cv2
import numpy as np
import base64
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Create "uploads" folder if not exists
UPLOAD_FOLDER = "static/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load OpenCV Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    image_data = data['image']

    # Convert Base64 to OpenCV Image
    image_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Face Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"message": "No Face Detected!"})

    # Save Image
    image_name = f"selfie_{int(cv2.getTickCount())}.png"
    image_path = os.path.join(UPLOAD_FOLDER, image_name)
    cv2.imwrite(image_path, img)

    return jsonify({"message": "Image Saved Successfully!", "image": image_name})

if __name__ == '__main__':
    app.run(debug=True)
