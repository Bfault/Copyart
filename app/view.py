import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import imghdr #deprecated
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from matplotlib import artist
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

from run_model import transform_image
from data import ARTISTS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1048 * 1048
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'app/static/uploads/'

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files, artists=ARTISTS)

@app.route('/uploads', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return render_template('index.html', filename=filename, artists=ARTISTS)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return redirect(url_for('index'))

@app.route('/transform', methods=['POST'])
def transform():
    artist_name = request.form['artist']
    uploaded_file = request.files['file']

    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        path = f"{app.config['UPLOAD_PATH']}/{filename}"
        uploaded_file.save(path)
        result = transform_image(path, artist_name)
        print(type(result))
        plt.imsave(f"{app.config['UPLOAD_PATH']}/result_{filename}", result)
    return render_template('index.html', filename=filename, result=f"result_{filename}", artists=ARTISTS)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)