from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

app = Flask(__name__)

# YOLO for person detection
yolo_model = YOLO('yolov8n.pt')

# Load face/person classifier
face_model = load_model('realtime_identification.h5')

# Updated to match the model's trained input resolution
# Inspect model.input_shape if your model was trained on a size other than 128x128
IMG_SIZE = (128, 128) 

CLASS_NAMES = ['gieo', 'nat', 'jb', 'joshua', 'jeoff']  # Must match training class_indices

def classify_face(crop):
    img = cv2.resize(crop, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    preds = face_model.predict(img, verbose=0)
    idx = np.argmax(preds[0])
    confidence = preds[0][idx]
    return CLASS_NAMES[idx], confidence

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            results = yolo_model(frame, classes=[0], verbose=False)  # class 0 = person

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Prevent invalid cropping boundaries
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue

                try:
                    name, conf = classify_face(crop)
                    label = f"{name} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Classification error: {e}")
                    continue

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)