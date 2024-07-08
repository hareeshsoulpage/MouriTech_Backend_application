from flask import Flask, render_template, Response, request,jsonify
from flask_cors import CORS
import cv2
import torch
from PIL import Image
from super_gradients.training import models
from deepface import DeepFace
import base64
import numpy as np
app = Flask(__name__)
CORS(app)

# Set the device to GPU if available, otherwise fallback to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLO-NAS model
yolo_model = models.get(
    model_name='yolo_nas_s',  # specify the model name herepytho
    num_classes=2,
    checkpoint_path='model/average_model.pth'
).to(DEVICE)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    print("inside video_feed")
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = base64.b64decode(image_data)
    np_arr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Process YOLO-NAS object detection on every 5th frame
    # Convert frame to PIL image for YOLO-NAS object detection
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform YOLO-NAS object detection
    yolo_results = yolo_model.predict([img], conf=0.40)
    yolo_boxes = yolo_results.prediction.bboxes_xyxy
    yolo_labels = yolo_results.prediction.labels
    yolo_scores = yolo_results.prediction.confidence
    class_names = ['Safety Helmet', 'Reflective Jacket'] 

    for box, label, score in zip(yolo_boxes, yolo_labels, yolo_scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_names[label]}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': encoded_image})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame',methods=['POST'])
def process_frame():
    return generate_frames()

if __name__ == '__main__':
    app.run(debug=True)
