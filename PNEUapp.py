from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Path to the model
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

@app.route('/pneumonia_detection', methods=['GET', 'POST'])
def pneumonia_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = Image.open(filepath)
            processed_img = preprocess_image(img)

            # Model prediction
            prediction = model.predict(processed_img)
            prediction = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'

            # Pass the image and prediction to result.html
            image_url = url_for('static', filename=f'images/{filename}')
            return render_template('result.html', image_url=image_url, prediction=prediction)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        img = Image.open(file)
        processed_img = preprocess_image(img)
        
        # Model prediction
        prediction = model.predict(processed_img)
        prediction = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'
        
        # Render result page with prediction
        return render_template('result.html', prediction=prediction)


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')  # Store images in the static folder
    print(target)

    if not os.path.isdir(target):
        os.makedirs(target)

    file = request.files['file']
    filename = file.filename
    dest = os.path.join(target, filename)
    file.save(dest)

    # Call the prediction function
    status = check(dest)

    # Generate the URL for displaying the image
    image_url = url_for('static', filename=f'images/{filename}')

    return render_template('complete.html', image_url=image_url, prediction=status)


if __name__ == '__main__':
    app.run(debug=True)
