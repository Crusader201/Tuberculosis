from flask import Flask, render_template

app = Flask(__name__,template_folder='templates')

models = [
    {'name': 'DenseNet',
     },
    {'name': 'MobileNet',
     },
    {'name': 'XceptionNet',
     },
    {'name': 'NasNetLarge',
     },
    {'name': 'InceptionNet',
     },
]

placeholder = """
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
"""

@app.route('/')
def index():
    return render_template("homepage.html",models=models,placeholder=placeholder)

@app.route('/model/<str:name>')
def model(name):
    return render_template("model.html",model=models[name])

if __name__ == "__main__":
    app.run(debug=True)