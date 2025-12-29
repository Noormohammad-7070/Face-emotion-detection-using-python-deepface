from flask import Flask, render_template, request
import cv2
import os
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            # Read image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Analyze emotion
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(analysis, list):
                emotion = analysis[0]["dominant_emotion"]
            else:
                emotion = analysis["dominant_emotion"]

            # Draw face box and emotion
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            # Save output image
            cv2.imwrite(image_path, img)

    return render_template("index.html", image=image_path, emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
