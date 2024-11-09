from flask import Flask, send_from_directory
import json

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/training_status')
def training_status():
    try:
        with open('static/training_status.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'error': 'Training has not started yet'}

@app.route('/plot')
def get_plot():
    try:
        with open('static/plot.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'error': 'No plot available yet'}

@app.route('/test_samples')
def get_test_samples():
    try:
        with open('static/test_samples.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'error': 'No test samples available yet'}

if __name__ == '__main__':
    app.run(debug=True) 