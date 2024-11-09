# MNIST Training Monitor

A real-time web-based monitoring system for training a CNN on the MNIST dataset. The system shows live training progress, loss curves, and sample predictions through a web interface.

## Project Structure 
project/
├── model.py # CNN model architecture
├── train.py # Training script with monitoring
├── server.py # Flask server for web interface
├── static/ # Static files directory
│ ├── index.html # Web interface
│ ├── training_status.json # Current training status
│ ├── plot.json # Training plots
│ └── test_samples.json # Test predictions
└── mnist_cnn.pth # Saved model weights (after training)


## Requirements

1. Python 3.8+
2. PyTorch
3. Flask
4. Matplotlib
5. tqdm

Install dependencies:
```bash
pip install torch torchvision flask matplotlib tqdm
```

## Hardware Requirements

- GPU with CUDA support (recommended)
- Alternatively works with:
  - Apple Silicon (MPS)
  - CPU (slower training)

## Setup and Execution

1. Create project directory and files:
```bash
mkdir session_04_mnist_monitor
cd session_04_mnist_monitor
mkdir static
```

2. Copy all provided files into their respective locations:
- `model.py` - CNN architecture
- `train.py` - Training script
- `server.py` - Web server
- `static/index.html` - Web interface

3. Start the Flask server (in one terminal):
```bash
python server.py
```

The server will start at `http://localhost:5000`

4. Start training (in another terminal):
```bash
python train.py
```


5. Open web browser and navigate to: (or displayed address)
```
http://localhost:5000
```


## Features

### Model Architecture
- 4-layer CNN with:
  - 4 Convolutional layers
  - 2 Max pooling layers
  - 2 Dropout layers
  - 2 Fully connected layers
  - Batch size: 512
  - Learning rate: 0.01
  - Epochs: 10

### Real-time Monitoring
The web interface shows:
- Current training status
  - Epoch number
  - Batch number
  - Current loss
  - Current accuracy
  - Overall progress
- Live-updating plots
  - Training loss curve
  - Test accuracy curve
- Sample predictions on test data (after training)

### Training Logs
Console output shows:
- Device information (CUDA/MPS/CPU)
- GPU details (if available)
- Training parameters
- Progress bar with:
  - Overall progress
  - Current epoch
  - Loss
  - Accuracy
- Epoch summaries with:
  - Training loss
  - Training accuracy
  - Test loss
  - Test accuracy

## File Descriptions

### model.py
Contains the CNN architecture (MNISTNet) with:
- Input: 28x28 grayscale images
- 4 convolutional layers
- Dropout for regularization
- Output: 10 classes (digits 0-9)

### train.py
Handles:
- Dataset loading
- Training loop
- Progress monitoring
- Web interface updates
- Model evaluation
- Plot generation

### server.py
Flask server providing:
- Web interface serving
- JSON endpoints for:
  - Training status
  - Loss plots
  - Test samples

### static/index.html
Web interface with:
- Training status display
- Loss/accuracy plots
- Test sample predictions
- Auto-updating components

## Notes

1. The system automatically detects available hardware:
   - CUDA GPU (preferred)
   - Apple Silicon MPS
   - CPU (fallback)

2. Training artifacts are saved in:
   - `static/` directory (monitoring files)
   - `mnist_cnn.pth` (trained model)

3. The web interface updates:
   - Status: Every 1 second
   - Plots: Every 5 seconds
   - Test samples: Every 5 seconds

4. For better performance:
   - Use GPU if available
   - Adjust batch size based on available memory
   - Modify update frequencies if needed