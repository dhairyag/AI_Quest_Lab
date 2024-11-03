from nltk.corpus import wordnet
import random
from typing import List, Union, Tuple, Dict, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
import librosa
import soundfile as sf
import math

def apply_augmentation(data: Union[str, np.ndarray, Dict], data_type: str, technique: str) -> Union[str, np.ndarray, Dict]:
    """Applies a data augmentation technique."""
    
    if data_type == "text":
        if technique == "Synonym Replacement":
            return synonym_replacement(data)
        elif technique == "Random Insertion":
            return random_insertion(data)
        else:
            raise ValueError(f"Unsupported augmentation technique for text: {technique}")
            
    elif data_type == "image":
        # Convert base64 string back to numpy array if needed
        if isinstance(data, str):
            img_data = base64.b64decode(data)
            data = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
        # Convert BGR to RGB
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        if technique == "Flip":
            augmented = apply_flip(pil_image)
        elif technique == "Rotate":
            augmented = apply_rotation(pil_image)
        elif technique == "Add Noise":
            augmented = apply_color_jitter(pil_image)
        else:
            raise ValueError(f"Unsupported augmentation technique for image: {technique}")
            
        # Convert back to BGR for OpenCV
        numpy_image = np.array(augmented)
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    elif data_type == "audio":
        try:
            # Convert audio data from frontend format to processing format
            if isinstance(data, dict) and 'audio_data' in data:
                samples, sample_rate = base64_to_audio(data['audio_data'])
            elif isinstance(data, tuple) and len(data) == 2:
                samples, sample_rate = data
            else:
                raise ValueError(f"Invalid audio data format. Got type: {type(data)}")

            # Ensure samples are numpy array and handle stereo to mono conversion
            samples = np.array(samples)
            if len(samples.shape) > 1:  # If stereo, convert to mono
                samples = np.mean(samples, axis=1)

            if technique == "Time Stretch":
                augmented_samples = apply_time_stretch(samples, sample_rate)
            elif technique == "Pitch Shift":
                augmented_samples = apply_pitch_shift(samples, sample_rate)
            elif technique == "Add Noise":
                augmented_samples = apply_background_noise(samples)
            else:
                raise ValueError(f"Unsupported augmentation technique for audio: {technique}")

            # Return in the same format as received
            return {
                'type': 'audio',
                'audio_data': audio_to_base64(augmented_samples, sample_rate),
                'sample_rate': sample_rate
            }
            
        except Exception as e:
            print(f"Error in audio augmentation: {str(e)}")
            raise ValueError(f"Error in audio augmentation: {str(e)}")
    elif data_type == "3d_geometry":
        try:
            # Ensure data contains required components
            if not isinstance(data, dict) or 'vertices' not in data or 'faces' not in data:
                raise ValueError("Invalid 3D geometry data format")
            
            vertices = data['vertices']
            faces = data['faces']
            
            if technique == "Random Rotation":
                augmented_data = apply_3d_random_rotation(vertices, faces)
            elif technique == "Random Scaling":
                augmented_data = apply_3d_random_scaling(vertices, faces)
            elif technique == "Add Noise":
                augmented_data = apply_3d_noise(vertices, faces)
            else:
                raise ValueError(f"Unsupported augmentation technique for 3D geometry: {technique}")
            
            # Generate visualization for the augmented geometry
            from utils import visualize_3d_geometry
            img_str = visualize_3d_geometry(
                np.array(augmented_data['vertices']), 
                augmented_data['faces']
            )
            
            # Return in the expected format
            return {
                'vertices': augmented_data['vertices'],
                'faces': augmented_data['faces'],
                'projection': f'data:image/png;base64,{img_str}'
            }
            
        except Exception as e:
            print(f"Error in 3D geometry augmentation: {str(e)}")
            raise ValueError(f"Error in 3D geometry augmentation: {str(e)}")
    else:
        raise ValueError(f"Augmentation not implemented for data type: {data_type}")

# Image augmentation functions
def apply_flip(image: Image.Image) -> Image.Image:
    """Apply random horizontal flip to the image."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # p=1.0 ensures flip always happens
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    return transform(image)

def apply_rotation(image: Image.Image, degrees: int = 30) -> Image.Image:
    """Apply random rotation to the image."""
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=degrees),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    return transform(image)

def apply_color_jitter(image: Image.Image) -> Image.Image:
    """Apply color jittering (brightness, contrast, saturation, hue)."""
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    return transform(image)

# You could also create a combined augmentation
def apply_combined_augmentation(image: Image.Image) -> Image.Image:
    """Apply multiple augmentations in sequence."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    return transform(image)

# Existing text augmentation functions
def synonym_replacement(text: str, p: float = 0.1) -> str:
    """Replaces words with synonyms."""
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < p:
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            if len(synonyms) > 0:
                new_words.append(random.choice(synonyms))
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def random_insertion(text: str, p: float = 0.1) -> str:
    """Inserts random words into text."""
    words = text.split()
    new_words = words.copy()
    for _ in range(int(len(words) * p)):
        insert_position = random.randint(0, len(new_words))
        new_words.insert(insert_position, random.choice(words))
    return ' '.join(new_words)

# Audio augmentation functions
def apply_time_stretch(samples: np.ndarray, sample_rate: int, rate: float = 1.2) -> np.ndarray:
    """Apply time stretching to audio samples."""
    return librosa.effects.time_stretch(y=samples, rate=rate)

def apply_pitch_shift(samples: np.ndarray, sample_rate: int, steps: int = 2) -> np.ndarray:
    """Apply pitch shifting to audio samples."""
    return librosa.effects.pitch_shift(y=samples, sr=sample_rate, n_steps=steps)

def apply_background_noise(samples: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add random background noise to audio samples."""
    noise = np.random.randn(*samples.shape)
    augmented_samples = samples + noise_factor * noise
    return augmented_samples

def base64_to_audio(base64_str: str) -> Tuple[np.ndarray, int]:
    """Convert base64 string to audio samples and sample rate."""
    audio_data = base64.b64decode(base64_str)
    audio_io = io.BytesIO(audio_data)
    samples, sample_rate = sf.read(audio_io)
    return samples, sample_rate

def audio_to_base64(samples: np.ndarray, sample_rate: int) -> str:
    """Convert audio samples to base64 string."""
    audio_io = io.BytesIO()
    sf.write(audio_io, samples, sample_rate, format='WAV')
    audio_io.seek(0)
    return base64.b64encode(audio_io.read()).decode()

# Add these new functions for 3D geometry augmentation
def apply_3d_random_rotation(vertices: List[List[float]], faces: List[List[int]]) -> Dict[str, Any]:
    """Apply random rotation to 3D geometry vertices."""
    # Convert vertices to PyTorch tensor
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    
    # Generate random rotation angles for each axis
    angles = torch.rand(3) * 2 * math.pi
    
    # Create rotation matrices
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angles[0]), -torch.sin(angles[0])],
        [0, torch.sin(angles[0]), torch.cos(angles[0])]
    ])
    
    Ry = torch.tensor([
        [torch.cos(angles[1]), 0, torch.sin(angles[1])],
        [0, 1, 0],
        [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
    ])
    
    Rz = torch.tensor([
        [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
        [torch.sin(angles[2]), torch.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    # Apply combined rotation
    R = Rz @ Ry @ Rx
    rotated_vertices = vertices_tensor @ R
    
    return {
        'vertices': rotated_vertices.tolist(),
        'faces': faces
    }

def apply_3d_random_scaling(vertices: List[List[float]], faces: List[List[int]]) -> Dict[str, Any]:
    """Apply random scaling to 3D geometry vertices."""
    # Convert vertices to PyTorch tensor
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    
    # Generate random scaling factors for each axis
    scale_factors = torch.rand(3) * 0.5 + 0.75  # Random scale between 0.75 and 1.25
    
    # Apply scaling
    scaled_vertices = vertices_tensor * scale_factors
    
    return {
        'vertices': scaled_vertices.tolist(),
        'faces': faces
    }

def apply_3d_noise(vertices: List[List[float]], faces: List[List[int]], noise_factor: float = 0.02) -> Dict[str, Any]:
    """Add random noise to 3D geometry vertices."""
    # Convert vertices to PyTorch tensor
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    
    # Generate random noise
    noise = torch.randn_like(vertices_tensor) * noise_factor
    
    # Add noise to vertices
    noisy_vertices = vertices_tensor + noise
    
    return {
        'vertices': noisy_vertices.tolist(),
        'faces': faces
    }
