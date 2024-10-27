import torch
from typing import List, Union

def apply_preprocessing(data: Union[str, List[str]], data_type: str, step: str) -> Union[List[str], torch.Tensor]:
    """Applies a preprocessing step to the data."""
    
    if data_type == "text":
        if step == "Tokenize":
            return tokenize_text(data)
        elif step == "Pad":
            return pad_or_truncate(data)
        elif step == "Embed":
            embeddings = embed_tokens(tokenize_text(data))
            # return string of embeddings
            return embeddings.tolist()
        elif step == 'Remove Punctuation':
            import string
            return data.translate(str.maketrans('', '', string.punctuation))
        else:
            raise ValueError(f"Unsupported preprocessing step for text: {step}")
    else:
        raise ValueError(f"Preprocessing not implemented for data type: {data_type}")

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
