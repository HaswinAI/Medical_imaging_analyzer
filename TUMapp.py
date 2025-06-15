from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Trained Model
MODEL_PATH = "model/best_brain_tumor_model.keras"  # Replace with your model's path
model = load_model(MODEL_PATH)

# Function to process the image
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)
    return prediction[0][0] > 0.5  # Assuming binary classification (Tumor/No Tumor)

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        prediction = model_predict(filepath, model)
        return render_template("result.html", image_name=file.filename, predvalue=prediction)

if __name__ == "__main__":
    app.run(debug=True)
