from nltk.corpus import wordnet
import random
from typing import List, Union
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64

def apply_augmentation(data: Union[str, np.ndarray], data_type: str, technique: str) -> Union[str, np.ndarray]:
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
