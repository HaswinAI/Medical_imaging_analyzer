from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Path to store uploaded images
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_path = os.path.join(app.root_path, 'model', 'model.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = load_model(model_path)

def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match input size of model
    img = img.convert('RGB')  # Convert to RGB if necessary
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    prediction = None
    prediction_color = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('pneumonia'))

        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('pneumonia'))

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = Image.open(filepath)
            processed_img = preprocess_image(img)

            # Model prediction
            pred_prob = model.predict(processed_img)[0][0]

            # ✅ Assign text and color
            if pred_prob > 0.5:
                prediction = "PNEUMONIA DETECTED"
                prediction_color = "red-text"  # Class for red color
            else:
                prediction = "NORMAL LUNG"
                prediction_color = "green-text"  # Class for green color

            # Image URL for rendering
            image_url = url_for('static', filename=f'images/{filename}')

    return render_template('pneumonia.html', prediction=prediction, prediction_color=prediction_color, image_url=image_url)


@app.route("/brain-tumor", methods=['GET', 'POST'])
def brain_tumor_detection():
    prediction_text = None
    prediction_color = None
    image_url = None

    if request.method == 'POST':
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = Image.open(filepath)
            processed_img = preprocess_image(img)  # Using the same preprocessing function

            # Model prediction
            pred_prob = model.predict(processed_img)[0][0]

            #  Assign text and color
            if pred_prob > 0.5:
                prediction_text = "BRAIN TUMOR DETECTED"
                prediction_color = "red-text"  # Class for red color
            else:
                prediction_text = "NO BRAIN TUMOR"
                prediction_color = "green-text"  # Class for green color

            image_url = url_for('static', filename=f'images/{filename}')

    return render_template("brain-tumor.html", brain_prediction=prediction_text, prediction_color=prediction_color, brain_image_url=image_url)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/cancer-detection')
def cancer_detection():
    return render_template('cancer-detection.html')


if __name__ == '__main__':
    app.run(debug=Falsr,host='0,0,0,0')
