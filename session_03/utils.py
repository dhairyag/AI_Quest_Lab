import librosa # Example for audio
import cv2 # Example for image
import torch
from typing import Any
import io
import numpy as np
import base64
import soundfile as sf

def load_data(file_content: bytes, data_type: str) -> Any:
    """Loads data based on the specified type."""
    if data_type == "text":
        return file_content.decode("utf-8")
    elif data_type == "audio":
        return librosa.load(io.BytesIO(file_content))
    elif data_type == "image":
        return cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def show_sample(data, data_type: str) -> Any:
    """Returns a sample of data for display (consider data type)."""
    if data_type == "text":
        return data[:5000] + "... [only showing first 5000 characters]" if len(data) > 5000 else data
    elif data_type == "audio":
        # Convert audio data to base64 for frontend playback
        buffer = io.BytesIO()
        sf.write(buffer, data[0], data[1], format='wav')
        return base64.b64encode(buffer.getvalue()).decode()
    elif data_type == "image":
        # Convert image to base64 for frontend display
        _, buffer = cv2.imencode('.png', data)
        return base64.b64encode(buffer).decode()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
