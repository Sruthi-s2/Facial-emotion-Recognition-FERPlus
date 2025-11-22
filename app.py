import os
import cv2
import numpy as np
import shutil
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = (112, 112)
model = load_model("best_model.h5")
class_names = np.load("labels.npy", allow_pickle=True).tolist()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

def predict_emotion(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Invalid image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return "No face detected"

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(face, IMG_SIZE)
    face_input = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_input, axis=(0, -1))

    pred = model.predict(face_input)
    pred_idx = int(np.argmax(pred))

    if pred_idx >= len(class_names):
        return "Prediction error"

    return class_names[pred_idx]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file uploaded"

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        static_path = os.path.join(STATIC_FOLDER, file.filename)
        shutil.copy(file_path, static_path)

        prediction = predict_emotion(file_path)
        return render_template("index.html", filename=file.filename, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
