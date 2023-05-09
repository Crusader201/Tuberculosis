from flask import Flask, render_template, request
from keras.models import load_model
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates')

models = [
    {'name': 'DenseNet',
     'Accuracy': '28.78',
     'Precision': '19.20',
     'Recall': '27.60',
     'FMeasure': '22.59'
     },
    {'name': 'MobileNet',
     'Accuracy': '78.04',
     'Precision': '79.53',
     'Recall': '76.87',
     'FMeasure': '75.74'
     },
    {'name': 'XceptionNet',
     'Accuracy': '95.12',
     'Precision': '94.79',
     'Recall': '94.72',
     'FMeasure': '94.74'
     },
    {'name': 'NasNetLarge',
     'Accuracy': '96.58',
     'Precision': '96.32',
     'Recall': '96.33',
     'FMeasure': '96.32'
     },
    {'name': 'InceptionNet',
     'Accuracy': '98.53',
     'Precision': '98.51',
     'Recall': '98.41',
     'FMeasure': '98.46'
     },
]

placeholder = """
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
"""


@app.route('/')
def index():
    return render_template("homepage.html", models=models, placeholder=placeholder)


@app.route('/model/<string:name>')
def model(name):
    for model in models:
        if model['name'] == name:
            return render_template("model.html", model=model)


@app.route('/testing',methods=["GET","POST"])
def test():
    if request.method == "GET":
        return render_template("testing.html")

    else:
        print("IN POST")
        file = request.files['file']
        fp = file.save('temp.jpg')
        img = cv2.imread('temp.jpg') #read images
        img = cv2.resize(img, (80,80)) #resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(80,80,3)
        print(im2arr.shape)
        model = load_model("models/inception.hdf5")
        predictions = model.predict(im2arr.reshape(-1,80,80,3))
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        labels = ['COVID-19', 'Normal', 'TB']
        img = cv2.resize(im2arr, (1000,600)) # increase the image size
        cv2.putText(img, 'Prediction Output : '+labels[predictions[0]]+" Detected.", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # increase font size and reposition text
        plt.imsave('static/temp.jpg',img)
        return {"success":True}, 201
        



if __name__ == "__main__":
    app.run(debug=True)
