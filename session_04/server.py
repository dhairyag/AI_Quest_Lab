from flask import Flask, send_from_directory, request, jsonify
import json
from threading import Thread
from train import train_models  # Import the train_models function

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
    data = request.json
    config1 = data['config1']
    config2 = data['config2']
    
    # Start training in a separate thread
    thread = Thread(target=train_models, args=(config1, config2))
    thread.daemon = True  # Make thread daemon so it dies when main thread dies
    thread.start()
    
    return jsonify({'status': 'Training started'})

@app.route('/model_info/<model_name>')
def get_model_info(model_name):
    try:
        with open(f'static/model_info_{model_name}.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'error': 'Model information not available yet'}

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    
    # Extract the new training parameters
    optimizer = data.get('optimizer', 'adam')
    batch_size = data.get('batch_size', 512)
    epochs = data.get('epochs', 10)
    
    # Pass these parameters to your training function
    training_config = {
        'optimizer': optimizer,
        'batch_size': batch_size,
        'epochs': epochs,
        # ... other existing config parameters ...
    }
    
    # Update your training call
    train_model(training_config)
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader when using threads