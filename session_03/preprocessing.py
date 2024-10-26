import nltk
import torch
from typing import List, Union

def apply_preprocessing(data: Union[str, List[str]], data_type: str, step: str) -> Union[List[str], torch.Tensor]:
    """Applies a preprocessing step to the data."""
    
    if data_type == "text":
        if step == "tokenize":
            return tokenize_text(data)
        elif step == "pad":
            return pad_sequences(data)
        elif step == "embed":
            return embed_text(data)
        else:
            raise ValueError(f"Unsupported preprocessing step for text: {step}")
    else:
        raise ValueError(f"Preprocessing not implemented for data type: {data_type}")

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text data."""
    return nltk.word_tokenize(text)

def pad_sequences(sequences: List[List[str]], max_len: int = 100) -> torch.Tensor:
    """Pads sequences to a fixed length."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + ['<PAD>'] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return torch.tensor([[1 if token != '<PAD>' else 0 for token in seq] for seq in padded_sequences])

def embed_text(tokens: List[str], embedding_dim: int = 100) -> torch.Tensor:
    """Maps tokens to vectors using a simple random embedding."""
    vocab = list(set(token for seq in tokens for token in seq))
    embedding = torch.randn(len(vocab), embedding_dim)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return torch.stack([torch.mean(torch.stack([embedding[word_to_idx[token]] for token in seq]), dim=0) for seq in tokens])

# Add more preprocessing functions for different data types and techniques
