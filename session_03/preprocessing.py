import torch
from typing import List, Union, Any
import torchvision.transforms as transforms
from PIL import Image
import librosa
from typing import Tuple, Dict
import json

import base64
import io
import numpy as np
import soundfile as sf
import librosa
import cv2  # If using MFCC visualization
import matplotlib.pyplot as plt
from utils import visualize_3d_geometry

def audio_to_base64(samples: np.ndarray, sample_rate: int) -> str:
    """
    Convert audio samples to base64 string.
    
    Args:
        samples: Audio samples
        sample_rate: Sample rate of the audio
    
    Returns:
        Base64 encoded string of the audio
    """
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format='wav')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_audio(audio_base64: str) -> Tuple[np.ndarray, int]:
    """
    Convert base64 string to audio samples and sample rate.
    
    Args:
        audio_base64: Base64 encoded audio string
    
    Returns:
        Tuple of (samples, sample_rate)
    """
    audio_bytes = base64.b64decode(audio_base64)
    audio_file = io.BytesIO(audio_bytes)
    samples, sample_rate = sf.read(audio_file)
    return samples, sample_rate

def _get_audio_data(data): # Helper function to consolidate data retrieval
    if isinstance(data, dict) and 'audio_data' in data:
        samples, sample_rate = base64_to_audio(data['audio_data'])
    elif isinstance(data, tuple) and len(data) == 2:
        samples, sample_rate = data
    else:
        raise ValueError("Invalid audio data format")
    return np.array(samples), sample_rate # Ensure numpy array

def normalize_audio(samples: np.ndarray, method="peak") -> np.ndarray:
    """
    Normalize audio using peak or RMS normalization.
    
    Args:
        samples: Input audio samples
        method: Normalization method ('peak' or 'rms')
    
    Returns:
        Normalized audio samples
    """
    if method == "peak":
        max_abs_value = np.abs(samples).max()
        if max_abs_value > 0:  # Avoid division by zero
            return samples / max_abs_value
        return samples
    elif method == "rms":
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0:
            return samples / rms
        return samples
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def resample_audio(samples, sample_rate, target_sr=16000):
    """Resamples audio to a specified target sample rate."""
    return librosa.resample(y=samples, orig_sr=sample_rate, target_sr=target_sr)

def calculate_mfcc(samples, sample_rate, n_mfcc=40, n_fft=2048, hop_length=512):
    """Calculates MFCC features."""
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc

def generate_spectrogram(samples, sample_rate, n_fft=2048, hop_length=512, window='hann', visualize=False):
    """Generates a spectrogram and optionally visualizes it."""
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, window=window)
    spectrogram = np.abs(stft)**2 # Power spectrogram
    if visualize:
        mel_spectrogram = librosa.feature.mel_spectrogram(S=spectrogram, sr=sample_rate)  # Mel scale for visualization
        db_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        spectrogram_image = librosa.display.specshow(db_mel_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        _, buffer = cv2.imencode('.png', spectrogram_image)
        image_base64 = base64.b64encode(buffer).decode()
        return {'type':'image', 'image_data': image_base64}

    return spectrogram  # Return numerical spectrogram

def trim_silence(samples, top_db=20):
    """Trims leading and trailing silence from audio."""
    return librosa.effects.trim(samples, top_db=top_db)


def apply_preprocessing(data: Any, data_type: str, step: str) -> Any:
    """Applies a preprocessing step to the data."""
    
    if data_type == "text":
        if step == "Tokenize":
            return tokenize_text(data)
        elif step == "Pad":
            tokens = tokenize_text(data)
            return pad_or_truncate(tokens)
        elif step == "Embed":
            tokens = tokenize_text(data)
            padded = pad_or_truncate(tokens)
            embeddings = embed_tokens(padded)
            return embeddings.tolist()
        elif step == 'Remove Punctuation':
            import string
            return data.translate(str.maketrans('', '', string.punctuation))
        else:
            raise ValueError(f"Unsupported preprocessing step for text: {step}")
            
    elif data_type == "image":
        # Convert base64 string back to numpy array if needed
        if isinstance(data, str):
            import base64
            img_data = base64.b64decode(data)
            data = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
        if step == "Resize":
            processed = resize_image(data)
        elif step == "Normalize":
            processed = normalize_image(data)
        elif step == "Grayscale":
            processed = convert_to_grayscale(data)
        elif step == "ML Preprocess":
            # Use the combined transform
            pil_image = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
            transform = get_ml_transform(grayscale=False)
            processed = transform(pil_image)
            # Convert tensor to numpy for display
            processed = processed.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unsupported preprocessing step for image: {step}")
            
        return processed
    elif data_type == "audio":
        try:
            # Import required modules at the start of audio processing
            import base64
            import io
            import soundfile as sf
            import matplotlib.pyplot as plt
            import librosa.display
            
            # Convert audio data from frontend format to processing format
            if isinstance(data, dict) and 'audio_data' in data:
                # For MFCC, we need to decode the base64 audio data first
                samples, sample_rate = base64_to_audio(data['audio_data'])
            elif isinstance(data, tuple) and len(data) == 2:
                samples, sample_rate = data
            else:
                raise ValueError(f"Invalid audio data format. Got type: {type(data)}")

            # Ensure samples are numpy array and handle stereo to mono conversion
            samples = np.array(samples)
            if len(samples.shape) > 1:  # If stereo, convert to mono
                samples = np.mean(samples, axis=1)

            # Default parameters for each processing step
            params = {
                "Normalize": {"method": "peak"},
                "Resample": {"target_sr": 16000},
                "MFCC": {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512},
                "Trim Silence": {"top_db": 20}
            }

            if step == "Normalize":
                processed_samples = normalize_audio(samples, **params["Normalize"])
                return {
                    'type': 'audio',
                    'audio_data': audio_to_base64(processed_samples, sample_rate),
                    'sample_rate': sample_rate
                }
            
            elif step == "MFCC":
                print(f"Processing MFCC with samples shape: {samples.shape}, sample_rate: {sample_rate}")
                
                try:
                    # Calculate MFCC features
                    mfcc_features = librosa.feature.mfcc(
                        y=samples, 
                        sr=sample_rate,
                        **params["MFCC"]
                    )
                    
                    print(f"MFCC features shape: {mfcc_features.shape}")
                    
                    # Create figure and plot MFCC
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(
                        mfcc_features,
                        sr=sample_rate,
                        hop_length=params["MFCC"]["hop_length"],
                        x_axis='time',
                        y_axis='mel'
                    )
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('MFCC Spectrogram')
                    
                    # Save plot to bytes buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    plt.close()  # Close the figure to free memory
                    buf.seek(0)
                    
                    # Convert to base64
                    img_base64 = base64.b64encode(buf.getvalue()).decode()
                    
                    # Return in a format that show_sample can handle directly
                    # This is different from other audio preprocessing steps
                    # as it returns a spectrogram image instead of audio data
                    return {
                        'type': 'spectrogram',  # Explicit type for frontend handling
                        'image_data': img_base64,
                        'description': f'MFCC Spectrogram ({params["MFCC"]["n_mfcc"]} coefficients)',
                        'sample_rate': sample_rate,  # Keep sample rate for reference
                        'is_mfcc': True  # Flag to identify MFCC specifically
                    }
                    
                except Exception as e:
                    print(f"Error in MFCC calculation: {str(e)}")
                    raise ValueError(f"MFCC calculation failed: {str(e)}")

            elif step == "Resample":
                processed_samples = resample_audio(samples, sample_rate, **params["Resample"])
                new_sample_rate = params["Resample"]["target_sr"]
                return {
                    'type': 'audio',
                    'audio_data': audio_to_base64(processed_samples, new_sample_rate),
                    'sample_rate': new_sample_rate
                }
                
            elif step == "Spectrogram":
                result = generate_spectrogram(samples, sample_rate, **params["Spectrogram"])
                if isinstance(result, dict):
                    return result
                else:
                    spectrogram = ((result - result.min()) * 255 / 
                                 (result.max() - result.min())).astype(np.uint8)
                    spectrogram_colored = cv2.applyColorMap(spectrogram, cv2.COLORMAP_VIRIDIS)
                    _, buffer = cv2.imencode('.png', spectrogram_colored)
                    return {
                        'type': 'spectrogram',
                        'image_data': base64.b64encode(buffer).decode(),
                        'description': 'Power Spectrogram',
                        'sample_rate': sample_rate
                    }
                
            elif step == "Trim Silence":
                processed_samples, _ = trim_silence(samples, **params["Trim Silence"])
                return {
                    'type': 'audio',
                    'audio_data': audio_to_base64(processed_samples, sample_rate),
                    'sample_rate': sample_rate
                }
                
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            print(f"Input data type: {type(data)}")
            if isinstance(data, dict):
                print(f"Dictionary keys: {data.keys()}")
            raise ValueError(f"Error processing audio: {str(e)}")
    elif data_type == "3d_geometry":
        # Ensure data is in the correct format
        if not isinstance(data, dict) or 'vertices' not in data or 'faces' not in data:
            raise ValueError("Invalid 3D geometry data format")
            
        # Convert vertices to numpy array if not already
        vertices = np.array(data['vertices'])
        faces = data['faces']
        
        if step == "Normalize":
            # Scale to unit sphere
            center = vertices.mean(axis=0)
            vertices = vertices - center
            max_dist = np.max(np.linalg.norm(vertices, axis=1))
            if max_dist > 0:
                vertices = vertices / max_dist
            
        elif step == "Centering":
            # Center at origin
            center = vertices.mean(axis=0)
            vertices = vertices - center
        else:
            raise ValueError(f"Unsupported preprocessing step for 3D geometry: {step}")
            
        # Generate new visualization
        img_str = visualize_3d_geometry(vertices, faces)
        
        # Return consistent format
        return {
            'vertices': vertices.tolist(),
            'faces': faces,
            'projection': f'data:image/png;base64,{img_str}'
        }
        
    else:
        raise ValueError(f"Preprocessing not implemented for data type: {data_type}")

# Image preprocessing functions
def resize_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize image using PyTorch transforms.
    
    Args:
        image: Input image as numpy array
        target_size: Tuple of (height, width)
    
    Returns:
        Resized image as numpy array
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply resize transform
    resize_transform = transforms.Resize(target_size)
    resized_image = resize_transform(pil_image)
    
    # Convert back to numpy array
    return cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image using PyTorch transforms.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Normalized image as numpy array
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Define normalization transform
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Apply normalization
    normalized_tensor = normalize_transform(pil_image)
    
    # Convert back to numpy array
    return normalized_tensor.permute(1, 2, 0).numpy()

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale using PyTorch transforms.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Grayscale image as numpy array
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Define grayscale transform
    grayscale_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    # Apply transform
    gray_tensor = grayscale_transform(pil_image)
    
    # Convert back to numpy array and maintain 3D structure
    return gray_tensor.permute(1, 2, 0).numpy()

# You can also create a combined transform for common ML preprocessing
def get_ml_transform(target_size: tuple = (224, 224), grayscale: bool = False) -> transforms.Compose:
    """
    Get a standard ML preprocessing transform pipeline.
    
    Args:
        target_size: Desired image size
        grayscale: Whether to convert to grayscale
    
    Returns:
        PyTorch transform composition
    """
    transform_list = []
    
    # Add grayscale if requested
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    
    # Add standard transforms
    transform_list.extend([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406] if not grayscale else [0.5],
            std=[0.229, 0.224, 0.225] if not grayscale else [0.5]
        )
    ])
    
    return transforms.Compose(transform_list)

# Existing text preprocessing functions
from transformers import AutoTokenizer, AutoModel
import torch

# Choose a pre-trained model and tokenizer (e.g., BERT)
model_name = "bert-base-uncased"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
except OSError:
    # Download and save if not found locally
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    model.save_pretrained(model_name)

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text using the chosen tokenizer."""
    return tokenizer.tokenize(text) 

def pad_or_truncate(tokens: List[str], max_len: int = 16) -> List[str]:
    """Pads or truncates a sequence of tokens to a fixed length."""
    if len(tokens) < max_len:
        return tokens + [tokenizer.pad_token] * (max_len - len(tokens))
    else:
        return tokens[:max_len]

def embed_tokens(tokens: List[str]) -> torch.Tensor:
    """
    Maps tokens to vectors using the chosen pre-trained model.
    
    Returns: 
        A tensor of shape (max_len, embedding_dim), 
        where embedding_dim is determined by the model.
    """
    # First truncate tokens to BERT's maximum length (512)
    tokens = tokens[:512]
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])  

    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state

    return embeddings
