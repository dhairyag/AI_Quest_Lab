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
        # Use soundfile instead of librosa for consistency
        audio_file = io.BytesIO(file_content)
        samples, sample_rate = sf.read(audio_file)
        return (samples, sample_rate)
    elif data_type == "image":
        return cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def show_sample(data, data_type: str) -> Any:
    """Returns a sample of data for display (consider data type)."""
    if data_type == "text":
        return data[:5000] + "... [only showing first 5000 characters]" if len(data) > 5000 else data
    elif data_type == "audio":
        try:
            # Convert audio data to base64 for frontend playback
            buffer = io.BytesIO()
            
            # Ensure samples are float32
            samples, sample_rate = data
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples)
            samples = samples.astype(np.float32)
            
            # Normalize audio if needed
            if np.abs(samples).max() > 1.0:
                samples = samples / np.abs(samples).max()
            
            # Write to WAV format
            sf.write(buffer, samples, sample_rate, format='wav', subtype='FLOAT')
            buffer.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Log for debugging
            print(f"Audio data processed - Sample rate: {sample_rate}, Length: {len(samples)}")
            print(f"Base64 data length: {len(audio_base64)}")
            
            return {
                'audio_data': audio_base64,
                'sample_rate': sample_rate,
                'duration': len(samples) / sample_rate  # Add duration info
            }
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise
    elif data_type == "image":
        # If data is a tensor, convert to numpy array
        if isinstance(data, torch.Tensor):
            data = data.permute(1, 2, 0).numpy()
            
        # Convert to uint8 range if normalized
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = ((data + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
        # Convert to base64 for frontend display
        _, buffer = cv2.imencode('.png', data)
        return base64.b64encode(buffer).decode()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
