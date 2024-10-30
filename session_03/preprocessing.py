import torch
from typing import List, Union, Any
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

def apply_preprocessing(data: Union[str, np.ndarray, List[str]], data_type: str, step: str) -> Union[List[str], torch.Tensor, np.ndarray]:
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
