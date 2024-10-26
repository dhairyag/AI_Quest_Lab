from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any
from preprocessing import apply_preprocessing
from augmentation import apply_augmentation
from utils import load_data, show_sample
import json

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Data loading endpoint
@app.post("/upload/{data_type}")
async def upload_data(data_type: str, file: UploadFile = File(...)):
    """
    Uploads a data file.
    
    Args:
        data_type: Type of data (e.g., "text", "audio", "image").
        file: Uploaded data file.
    
    Returns:
        Dictionary with success message and sample.
    """
    try:
        file_content = await file.read()
        data = load_data(file_content, data_type)
        return {"message": "Data uploaded successfully!", "sample": show_sample(data, data_type)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Preprocessing endpoint
@app.post("/preprocess/{data_type}")
async def preprocess_data(data_type: str, data: Dict[str, Any], 
                          preprocessing_steps: Optional[List[str]] = None):
    """
    Applies preprocessing steps to the data.
    """
    try:
        processed_data = data["data"]
        for step in preprocessing_steps or []:
            processed_data = apply_preprocessing(processed_data, data_type, step)
        return {"message": "Data preprocessed successfully!", "sample": show_sample(processed_data, data_type)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Augmentation endpoint
@app.post("/augment/{data_type}")
async def augment_data(data_type: str, data: Dict[str, Any],
                       augmentation_techniques: Optional[List[str]] = None):
    """
    Applies data augmentation techniques.
    """
    try:
        augmented_data = data["data"]
        for technique in augmentation_techniques or []:
            augmented_data = apply_augmentation(augmented_data, data_type, technique)
        return {"message": "Data augmented successfully!", "sample": show_sample(augmented_data, data_type)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
