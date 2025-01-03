from flask import Flask, send_from_directory, request, jsonify
import json
from threading import Thread
from train import train_models
import os

app = Flask(__name__)

def cleanup_old_files():
    """Clean up old JSON files from static folder"""
    json_files = [
        'static/plot.json',
        'static/test_samples.json',
        'static/training_status_Model 1.json',
        'static/training_status_Model 2.json',
        'static/model_info_Model 1.json',
        'static/model_info_Model 2.json'
    ]
    
    for file_path in json_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing {file_path}: {str(e)}")

@app.route('/')
def index():
    cleanup_old_files()
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/training_status')
def training_status():
    try:
        # Try to read both model statuses
        status = {}
        try:
            with open('static/training_status_Model 1.json', 'r') as f:
                status['model1'] = json.load(f)
        except FileNotFoundError:
            status['model1'] = {'error': 'Model 1 training has not started yet'}
            
        try:
            with open('static/training_status_Model 2.json', 'r') as f:
                status['model2'] = json.load(f)
        except FileNotFoundError:
            status['model2'] = {'error': 'Model 2 training has not started yet'}
            
        return status
    except Exception as e:
        return {'error': str(e)}

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

@app.route('/start_training', methods=['POST'])
def start_training():
    # Clean up old files before starting new training
    cleanup_old_files()
    
    data = request.json
    config1 = data['config1']
    config2 = data['config2']
    
    # Start training in a separate thread
    thread = Thread(target=train_models, args=(config1, config2))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Training started'})

@app.route('/model_info/<model_name>')
def get_model_info(model_name):
    try:
        with open(f'static/model_info_{model_name}.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'error': 'Model information not available yet'}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader when using threads