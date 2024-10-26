import librosa # Example for audio
import cv2 # Example for image
import torch
from typing import Any
import io
import numpy as np

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

def show_sample(data, data_type: str) -> str:
    """Returns a sample of data for display (consider data type)."""
    if data_type == "text":
        return data[:100] + "..." if len(data) > 100 else data
    elif data_type == "audio":
        return f"Audio length: {len(data[0])} samples, Sample rate: {data[1]}"
    elif data_type == "image":
        return f"Image shape: {data.shape}, Dtype: {data.dtype}"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
