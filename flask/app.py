import flask
import sys
import os
import time
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import importlib
sys.path.append('.')
import utils
import random

app = Flask(__name__)

# Config
app.config['MODEL_CONFIG'] = 'config.config'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(),'flask/static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
sample_files = ['samples/{}.jpg'.format(i) for i in range(10)]
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Model
def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global net
    model_config = importlib.import_module(app.config['MODEL_CONFIG'])
    net = utils.getNet(model_config)

# Pages
@app.route('/')
def index():
    return render_template('index.html', imagename=random.choice(sample_files), ver=time.time())


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if allowed_file(file.filename):
            saved_filename = app.config['UPLOAD_FOLDER']+'/input.jpg'
            file.save(saved_filename)
        return render_template('index.html', imagename="uploads/input.jpg" , ver=time.time())
    else:
        return render_template('index.html', imagename=random.choice(sample_files), ver=time.time())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        imagename=request.form.get('imagename')
        imagepath = os.path.join("flask/static/",imagename)
        print(imagepath)
        r = utils.predict(net, imagepath)
        pred = [{'label': k, 'score': round(v*100,2)} for k, v in enumerate(r)]
        return render_template('index.html', imagename=imagename, ver=time.time(), pred=pred)


@app.route('/static/<file>')
def static_file(file):
    return app.send_static_file(file)


@app.route('/favicon.ico')
def icon():
    return app.send_static_file('favicon.ico')


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5108)