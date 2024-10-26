from nltk.corpus import wordnet
import random
from typing import List

def apply_augmentation(data: str, data_type: str, technique: str) -> str:
    """Applies a data augmentation technique."""
    
    if data_type == "text":
        if technique == "synonym_replacement":
            return synonym_replacement(data)
        elif technique == "random_insertion":
            return random_insertion(data)
        else:
            raise ValueError(f"Unsupported augmentation technique for text: {technique}")
    else:
        raise ValueError(f"Augmentation not implemented for data type: {data_type}")

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

# Add more augmentation functions for various data types and techniques
