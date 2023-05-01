from flask import Flask,render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No image selected')

    img = Image.open(request.files['image'])
    img = img.convert('L')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model = load_model('my_model1.h5')
    pred = model.predict(img)[0][0]

    if pred >= 0.5:
        prediction = 'Covid'
    else:
        prediction = 'Non-Covid'

    return render_template('result.html', prediction=prediction,pred=pred)
