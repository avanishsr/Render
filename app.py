import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
from PIL import Image
import io
import base64

# Initialize the Flask app
app = Flask(__name__)

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model (replace 'best.pt' with the path to your weights)
model = YOLO('bestupadated.pt')

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return "Welcome to the YOLOv8 Number Plate Detection API!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform detection on the uploaded image
    results = model(file_path)

    # Extract the bounding boxes for the number plate (if detected)
    for result in results:
        boxes = result.boxes.xyxy  # x1, y1, x2, y2 for bounding boxes
        if len(boxes) > 0:
            # Assuming the first box is the number plate
            box = boxes[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # Load the original image using OpenCV
            img = cv2.imread(file_path)
            # Crop the image to the detected number plate
            cropped_img = img[y1:y2, x1:x2]

            # Convert the cropped image to PIL format
            pil_image = Image.fromarray(cropped_img)
            img_io = io.BytesIO()
            pil_image.save(img_io, 'JPEG')
            img_io.seek(0)

            # Encode the image to Base64
            base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Return Base64 encoded image as JSON response
            return jsonify({"cropped_image": base64_img})

    return jsonify({"error": "No number plate detected"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
